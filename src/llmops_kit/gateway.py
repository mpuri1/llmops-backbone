"""Client for the LiteLLM gateway (gateway/config.yaml): one OpenAI-compatible endpoint for all projects.

    from llmops_kit.gateway import chat
    reply = chat([{"role": "user", "content": "Hi"}], model="fast", project="my-project")

Projects ask for an alias ("fast", "smart"; both fall back to "fallback"), not a provider model. Every call
names its project, which the gateway uses for budgets and the spend ledger. The call is traced with llm_span:
tokens, the model that answered (after any fallback) and the cost the gateway reports. Standard library only,
so projects don't need a provider SDK to use it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from llmops_kit.tracing import llm_span

DEFAULT_URL = "http://localhost:4000"


class GatewayError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"gateway returned {status}: {detail}")
        self.status, self.detail = status, detail


@dataclass
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    retries: int = 0
    fallbacks: int = 0


def chat(
    messages: list[dict],
    model: str = "fast",
    project: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 120,
    **params,
) -> Reply:
    project = project or os.environ.get("LLMOPS_PROJECT")
    if not project:
        raise ValueError("name the calling project (project= or LLMOPS_PROJECT)")
    url = (base_url or os.environ.get("LLMOPS_GATEWAY_URL", DEFAULT_URL)).rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": messages, "metadata": {"project": project}, **params}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or os.environ.get('LLMOPS_GATEWAY_KEY', '')}",
        },
    )
    with llm_span(model=model, provider="litellm-gateway") as call:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
                headers = response.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise GatewayError(exc.code, detail) from None
        usage = payload.get("usage") or {}
        cost = headers.get("x-litellm-response-cost")
        reply = Reply(
            text=payload["choices"][0]["message"].get("content") or "",
            # the deployment that answered, after any fallback; the body only repeats the alias
            model=headers.get("x-litellm-model-name") or payload.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=float(cost) if cost not in (None, "") else None,
            retries=int(headers.get("x-litellm-attempted-retries") or 0),
            fallbacks=int(headers.get("x-litellm-attempted-fallbacks") or 0),
        )
        call.record_usage(reply.input_tokens, reply.output_tokens)
        call.record_response(model=reply.model, finish_reason=payload["choices"][0].get("finish_reason"))
        call.record_cost(reply.cost_usd)
        call.span.set_attribute("llmops.project", project)
        call.span.set_attribute("llmops.gateway.retries", reply.retries)
        call.span.set_attribute("llmops.gateway.fallbacks", reply.fallbacks)
    return reply
