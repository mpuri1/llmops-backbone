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
| `src/llmops_kit/online_eval.py` | Online evaluation: reference-free response checks, a deterministic sampler, drift monitors, a failure queue that becomes golden cases for the offline gate, and a canary router with automatic rollback |
| `examples/online_evals/` | A generated support-ticket stream, a script that sends it through the gateway, a monitor simulation and an analysis script |
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

`json: true` accepts a response that is JSON or JSON inside a single code fence (models often add one); any other text around it fails. The online checks use the same parser.

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

## Online evals

The eval gate runs before a merge. `llmops_kit.online_eval` is the production half; it calls no model and works on the records your calls already produce (`id`, `alias`, `input`, `output`, `latency_s`, `cost_usd`).

- **Checks** (`check_response`): reference-free, so they run on every response: valid JSON, required keys, allowed values, length, banned strings. A response passes when every group passes.
- **Sampling** (`sampled`): a hash of the request id, so the same requests are judged on every replay. Send the sampled ones to a judge model of your choice.
- **Monitors**: `RateMonitor` alerts when the upper end of the 95% Wilson interval of a pass rate over the last `window` responses falls below `baseline - tolerance`; `MeanMonitor` alerts when the mean of a cost or latency window passes `baseline * factor`. Each fires once per episode.
- **Failure queue** (`FailureQueue`, `promote`): flagged responses wait for a reviewer; accepted ones are appended to a cases file in the eval gate's format, so a production failure becomes a regression test.
- **Canary** (`CanaryRouter`, `CanaryController`): sends a fraction of requests to a candidate alias and rolls it back to zero when the candidate's monitor alerts. The candidate's monitor starts empty, so it may judge after `min_n` responses (default 20).

```bash
LLMOPS_TRACING=off uv run pytest tests/test_online_eval.py
uv run python -m llmops_kit.online_eval monitor --records log.jsonl --spec spec.json   # exit 0 no alert, 1 alert, 2 bad input
uv run python -m llmops_kit.online_eval promote --queue queue.jsonl --golden cases.jsonl --decisions decisions.json
uv run python examples/online_evals/simulate.py --check                                  # the fast simulation matches RateMonitor
```

`examples/online_evals/` holds a generated ticket stream with known labels (`tickets.py`), `collect.py` (sends it through the gateway under three prompt versions and a model alias swap, and stores every response), `simulate.py` (the monitor on simulated pass/fail streams, no model) and `analyse.py` (replays the stored responses through the monitors, the failure-to-golden-set loop and a canary). The stream is generated, not production traffic.
