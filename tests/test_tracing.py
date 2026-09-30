import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llmops_kit import llm_span, prompt_hash, setup_tracing, traced

EXPORTER = InMemorySpanExporter()


@pytest.fixture
def spans():
    # One exporter for the session: the global provider can only be installed once.
    setup_tracing("test-service", exporter=EXPORTER, batch=False) if not _installed() else None
    EXPORTER.clear()
    yield EXPORTER


def _installed() -> bool:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    return isinstance(trace.get_tracer_provider(), TracerProvider)


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_traced_function_creates_span_with_kind_and_service(spans):
    @traced(kind="TOOL")
    def profile_table(name):
        return name.upper()

    assert profile_table("claims") == "CLAIMS"
    span = by_name(spans)["test_traced_function_creates_span_with_kind_and_service.<locals>.profile_table"]
    assert span.attributes["openinference.span.kind"] == "TOOL"
    assert span.resource.attributes["service.name"] == "test-service"


def test_traced_records_and_reraises_errors(spans):
    @traced(name="boom")
    def fail():
        raise ValueError("bad mapping")

    with pytest.raises(ValueError):
        fail()
    span = by_name(spans)["boom"]
    assert span.status.status_code.name == "ERROR"
    assert span.events[0].name == "exception"


def test_llm_span_uses_genai_conventions_and_hashes_the_prompt(spans):
    template = "Map column {column} to an ontology property."
    with llm_span(model="gemini-2.5-flash", provider="gcp.gemini", prompt_template=template,
                  prompt_name="map_column") as call:
        call.record_usage(input_tokens=120, output_tokens=30)
        call.record_response(model="gemini-2.5-flash-001", finish_reason="stop")
    span = by_name(spans)["chat gemini-2.5-flash"]
    a = span.attributes
    assert a["gen_ai.request.model"] == "gemini-2.5-flash"
    assert a["gen_ai.provider.name"] == "gcp.gemini"
    assert a["gen_ai.usage.input_tokens"] == 120 and a["gen_ai.usage.output_tokens"] == 30
    assert a["llmops.prompt.hash"] == prompt_hash(template)
    assert template not in str(dict(a))  # prompt text is never stored


def test_nested_spans_share_a_trace(spans):
    @traced(kind="AGENT")
    def agent():
        with llm_span(model="m", provider="p"):
            pass

    agent()
    finished = spans.get_finished_spans()
    assert len({s.context.trace_id for s in finished}) == 1
    child = next(s for s in finished if s.name == "chat m")
    assert child.parent is not None


def test_prompt_hash_is_stable_and_sensitive():
    assert prompt_hash("a") == prompt_hash("a")
    assert prompt_hash("a") != prompt_hash("a ")
    assert len(prompt_hash("anything")) == 12


def test_second_setup_extends_instead_of_silently_replacing(spans):
    second = InMemorySpanExporter()
    setup_tracing("another-service", exporter=second, batch=False)

    @traced(name="after-second-setup")
    def work():
        return 1

    work()
    assert "after-second-setup" in by_name(second)  # the new exporter receives spans
    assert "after-second-setup" in by_name(spans)   # and the original one still does


def test_tracing_off_skips_network_export_but_keeps_explicit_exporters(spans, monkeypatch):
    monkeypatch.setenv("LLMOPS_TRACING", "off")
    explicit = InMemorySpanExporter()
    setup_tracing("test-service", exporter=explicit, batch=False)

    @traced(name="in-ci")
    def work():
        return 1

    work()
    assert "in-ci" in by_name(explicit)
