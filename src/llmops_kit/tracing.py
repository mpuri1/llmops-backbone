"""OpenTelemetry tracing for LLM applications.

One call sets up tracing for a service; spans go over OTLP/HTTP to Arize Phoenix
locally (default http://localhost:6006/v1/traces) or to any OTLP endpoint, e.g.
Azure Monitor via a collector.

Model-call spans use the OpenTelemetry GenAI semantic conventions (gen_ai.*) and
also set `openinference.span.kind`, which Phoenix uses to group and render spans.
Prompt templates are recorded by content hash, never by content, so traces
identify the prompt version without storing prompt text or data.

    from llmops_kit import setup_tracing, traced, llm_span, prompt_hash

    setup_tracing("ontology-dimension-factory")

    @traced(kind="TOOL")
    def profile_table(name): ...

    with llm_span(model="gemini-2.5-flash", provider="gcp.gemini", prompt_template=TEMPLATE) as call:
        response = client.generate(...)
        call.record_usage(input_tokens=..., output_tokens=...)
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)
DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"
TRACER_NAME = "llmops_kit"
SpanKind = Literal["CHAIN", "TOOL", "LLM", "RETRIEVER", "AGENT", "EVALUATOR"]
F = TypeVar("F", bound=Callable[..., Any])


def setup_tracing(
    service_name: str,
    endpoint: str | None = None,
    exporter: SpanExporter | None = None,
    batch: bool = True,
) -> TracerProvider:
    """Install (or extend) the global tracer provider for `service_name`.

    OpenTelemetry allows the global provider to be set only once per process; a
    second set is ignored with a warning, which would silently drop exporters.
    So if an SDK provider is already installed, the new exporter is attached to it
    instead, and the original service name is kept.

    `exporter` overrides the OTLP exporter (tests pass an in-memory one).
    LLMOPS_TRACING=off skips the default OTLP exporter (e.g. in CI); an exporter
    passed explicitly is always attached, so tests still see their spans.
    """
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        provider = current
        existing = provider.resource.attributes.get("service.name")
        if existing != service_name:
            logger.warning("tracing already set up for %r; keeping it, not %r", existing, service_name)
    else:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)
    if exporter is None:
        if os.environ.get("LLMOPS_TRACING", "on").lower() == "off":
            return provider  # the switch disables network export only
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(
            endpoint=endpoint or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", DEFAULT_ENDPOINT)
        )
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    return provider


def prompt_hash(template: str) -> str:
    """Stable short identifier for a prompt template (first 12 hex chars of SHA-256)."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


def traced(name: str | None = None, kind: SpanKind = "CHAIN") -> Callable[[F], F]:
    """Wrap a function in a span. Exceptions are recorded and re-raised."""

    def decorator(fn: F) -> F:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(TRACER_NAME)
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("openinference.span.kind", kind)
                span.set_attribute("code.function", fn.__qualname__)
                span.set_attribute("code.namespace", fn.__module__)
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass
class LLMCall:
    """Handle yielded by `llm_span` for recording results of a model call."""

    span: Span

    def record_usage(self, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        if input_tokens is not None:
            self.span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            self.span.set_attribute("llm.token_count.prompt", input_tokens)
        if output_tokens is not None:
            self.span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            self.span.set_attribute("llm.token_count.completion", output_tokens)

    def record_cost(self, usd: float | None) -> None:
        if usd is not None:
            self.span.set_attribute("llmops.cost.usd", usd)

    def record_response(self, model: str | None = None, finish_reason: str | None = None) -> None:
        if model:
            self.span.set_attribute("gen_ai.response.model", model)
        if finish_reason:
            self.span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])


@contextmanager
def llm_span(
    model: str,
    provider: str,
    operation: str = "chat",
    prompt_template: str | None = None,
    prompt_name: str | None = None,
) -> Iterator[LLMCall]:
    """Span for one model call, named '<operation> <model>' per the GenAI conventions."""
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(f"{operation} {model}") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("gen_ai.operation.name", operation)
        span.set_attribute("gen_ai.provider.name", provider)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("llm.model_name", model)
        if prompt_template is not None:
            span.set_attribute("llmops.prompt.hash", prompt_hash(prompt_template))
        if prompt_name:
            span.set_attribute("llmops.prompt.name", prompt_name)
        try:
            yield LLMCall(span)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
