# llmops-backbone

Shared tooling for LLM projects: a model gateway, OpenTelemetry tracing for model, tool and agent calls, and a project template that wires them in.

## What's here

| Path | What it is |
|---|---|
| `src/llmops_kit/` | `setup_tracing`, `@traced`, `llm_span`, `prompt_hash`: OpenTelemetry tracing with GenAI semantic conventions, exported over OTLP to Arize Phoenix or any collector |
| `gateway/` | [LiteLLM](https://docs.litellm.ai/) proxy config: one OpenAI-compatible endpoint with model aliases (`fast`, `smart`, and `fallback`, which both fall back to), retries; a hook that enforces per-project daily budgets and writes a spend ledger |
| `src/llmops_kit/bedrock_proxy.py` | Optional local proxy that SigV4-signs requests to Bedrock's OpenAI-compatible endpoint so the gateway can serve the `bedrock-glm` alias (`uv run --extra bedrock python -m llmops_kit.bedrock_proxy`, with `BEDROCK_PROFILE`; the gateway then needs `BEDROCK_PROXY_URL=http://127.0.0.1:4010/v1`) |
| `src/llmops_kit/gateway.py` | Client for the gateway: tags each call with its project and traces tokens, cost, retries, fallbacks and the model that answered |
| `template/` | [copier](https://copier.readthedocs.io/) template for new projects: uv, ruff, pytest, `.env` handling, tracing wired in, CI workflow |
| `src/llmops_kit/evalgate.py` | Offline eval gate: scores stored model outputs against a case file, diffs a candidate against a baseline and exits non-zero on a regression; renders the diff as a PR comment |
| `evals/` | The gate's case file and the stored baseline and candidate outputs |
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

## Eval gate in CI

`.github/workflows/eval-gate.yml` runs on every pull request. It compares stored outputs, so it calls no model, needs no API key and costs nothing. It posts (and updates) one PR comment with the per-metric diff, and fails the check on a regression.

Files, all JSONL (one object per line):

- `evals/cases.jsonl`: `id`, `input`, and any of `must_include`, `must_not_include`, `json`, `max_words`
- `evals/baseline.jsonl`: `{"id", "output"}` for the current prompt
- `evals/candidate.jsonl`: the same for the prompt in the PR

Metrics, all in 0 to 1, higher is better: `pass_rate` (cases passing every check), `include_rate`, `exclude_rate`, `format_rate`.

Tolerance rule:

1. A metric fails when `baseline - candidate > tolerance` (absolute, default `0.05`).
2. The gate also fails when more than `--max-case-regressions` (default `0`) cases passed in the baseline and fail in the candidate.
3. A case without a candidate output fails. Improvements never fail the gate.

In CI the cases and the baseline are read from the base branch, so a PR cannot loosen its own gate. To change a prompt: regenerate the outputs with your model, replace `evals/candidate.jsonl`, and open the PR. Fork PRs get a read-only token, so they get the job summary instead of a comment.

Run it locally; the fixtures in `tests/fixtures/eval_gate/` hold one passing and one failing candidate:

```bash
uv run python -m llmops_kit.evalgate --cases tests/fixtures/eval_gate/cases.jsonl \
  --baseline tests/fixtures/eval_gate/baseline.jsonl --candidate tests/fixtures/eval_gate/candidate_fail.jsonl
echo $?   # 0 pass, 1 gate failed, 2 bad input
```

Prompt templates are recorded in traces by content hash (`llmops.prompt.hash`), never by content.
