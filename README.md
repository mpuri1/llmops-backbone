# llmops-backbone

Shared tooling for LLM projects: a model gateway, OpenTelemetry tracing for model, tool and agent calls, and a project template that wires them in.

## What's here

| Path | What it is |
|---|---|
| `src/llmops_kit/` | `setup_tracing`, `@traced`, `llm_span`, `prompt_hash`: OpenTelemetry tracing with GenAI semantic conventions, exported over OTLP to Arize Phoenix or any collector |
| `gateway/` | [LiteLLM](https://docs.litellm.ai/) proxy config: one OpenAI-compatible endpoint with model aliases (`fast`, `smart`, and `fallback`, which both fall back to), retries; a hook that enforces per-project daily budgets and writes a spend ledger |
| `src/llmops_kit/gateway.py` | Client for the gateway: tags each call with its project and traces tokens, cost, retries, fallbacks and the model that answered |
| `template/` | [copier](https://copier.readthedocs.io/) template for new projects: uv, ruff, pytest, `.env` handling, tracing wired in, CI workflow |
| `examples/trace_smoke.py` | Sends an agent span with nested tool and model-call spans to local Phoenix |

## Try it

```bash
uv sync
uv run pytest
uv run --with arize-phoenix phoenix serve      # separate terminal; UI at http://localhost:6006
uv run python examples/trace_smoke.py
```

## Run the gateway

```bash
cp .env.example .env                         # GOOGLE_API_KEY, LLMOPS_GATEWAY_KEY
uv sync --extra gateway
uv run --extra gateway litellm --config gateway/config.yaml --port 4000
```

```python
from llmops_kit.gateway import chat

reply = chat([{"role": "user", "content": "Hi"}], model="fast", project="my-project")
print(reply.text, reply.model, reply.cost_usd)
```

## New project from the template

```bash
uv run copier copy --trust template ../my-project
```

Prompt templates are recorded in traces by content hash (`llmops.prompt.hash`), never by content.
