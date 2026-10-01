# llmops-backbone

Shared tooling for LLM projects: OpenTelemetry tracing for model, tool and agent calls, and a project template that wires it in.

## What's here

| Path | What it is |
|---|---|
| `src/llmops_kit/` | `setup_tracing`, `@traced`, `llm_span`, `prompt_hash`: OpenTelemetry tracing with GenAI semantic conventions, exported over OTLP to Arize Phoenix or any collector |
| `template/` | [copier](https://copier.readthedocs.io/) template for new projects: uv, ruff, pytest, `.env` handling, tracing wired in, CI workflow |
| `examples/trace_smoke.py` | Sends an agent span with nested tool and model-call spans to local Phoenix |

## Try it

```bash
uv sync
uv run pytest
uv run --with arize-phoenix phoenix serve      # separate terminal; UI at http://localhost:6006
uv run python examples/trace_smoke.py
```

## New project from the template

```bash
uv run copier copy --trust template ../my-project
```

Prompt templates are recorded in traces by content hash (`llmops.prompt.hash`), never by content.
