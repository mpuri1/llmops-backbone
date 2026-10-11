"""Send the generated ticket stream through the gateway and store every response (resumable).

Runs (ticket indexes, alias, prompt):
    v1     tickets 0-599    fast   v1   the stable setting; also the clean control
    v2     tickets 300-599  fast   v2   format regression (a sentence before the JSON)
    v3     tickets 300-599  fast   v3   silent quality regression (definitions removed)
    smart  tickets 300-599  smart  v1   alias swapped to the costly model, 300-token cap (its reasoning ate the cap)
    smart2 tickets 300-599  smart  v1   the same swap with a 1,500-token cap: the cost regression proper
    judge  a sample of the responses above, graded by ``smart`` against the rubric (no label shown)

Needs the gateway running (see the README). Responses go to <out-dir>/responses.jsonl and judge verdicts to
<out-dir>/judge.jsonl; a rerun skips what is already there, so nothing is paid for twice.

    LLMOPS_TRACING=off uv run python examples/online_evals/collect.py --out-dir results/e2 --runs v1 v2 v3 smart judge
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tickets import JUDGE_PROMPT, PROMPTS, make_ticket

from llmops_kit.gateway import GatewayError, chat
from llmops_kit.online_eval import sampled

PROJECT = "llmops-online-evals"
RUNS = {
    "v1": ("fast", "v1", range(0, 600), 300),
    "v2": ("fast", "v2", range(300, 600), 300),
    "v3": ("fast", "v3", range(300, 600), 300),
    "smart": ("smart", "v1", range(300, 600), 300),
    "smart2": ("smart", "v1", range(300, 600), 1500),
}
JUDGE_RATE = 0.2


class Store:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        self.ids = {r["id"] for r in self.rows}

    def add(self, row: dict) -> None:
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.rows.append(row)
            self.ids.add(row["id"])


class Spend:
    def __init__(self, cap: float, start: float = 0.0):
        self.total, self.cap, self.lock = start, cap, threading.Lock()

    def add(self, cost) -> None:
        with self.lock:
            self.total += cost or 0.0
            if self.total > self.cap:
                raise SystemExit(f"stopping: spend {self.total:.4f} passed the cap {self.cap}")


def call(messages: list[dict], alias: str, max_tokens: int = 300) -> dict:
    last = ""
    for attempt in range(3):
        t0 = time.time()
        try:
            r = chat(messages, model=alias, project=PROJECT, temperature=0, max_tokens=max_tokens)
            return {
                "output": r.text,
                "latency_s": round(time.time() - t0, 3),
                "cost_usd": r.cost_usd,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "model": r.model,
                "error": None,
            }
        except (GatewayError, OSError) as e:
            last = str(e)[:300]
            time.sleep(2 * (attempt + 1))
    return {
        "output": "",
        "latency_s": None,
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model": None,
        "error": last,
    }


def run_responses(name: str, store: Store, spend: Spend, workers: int) -> None:
    alias, version, indexes, max_tokens = RUNS[name]
    todo = [i for i in indexes if f"{name}-{i}" not in store.ids]

    def one(i: int) -> None:
        t = make_ticket(i)
        res = call(
            [{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": t["text"]}], alias, max_tokens
        )
        store.add(
            {
                "id": f"{name}-{i}",
                "run": name,
                "ticket": i,
                "alias": alias,
                "prompt": version,
                "input": t["text"],
                **res,
            }
        )
        spend.add(res["cost_usd"])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    print(f"{name}: {len(todo)} new responses, spend so far ${spend.total:.4f}")


def run_judge(responses: Store, judged: Store, spend: Spend, workers: int) -> None:
    todo = [
        r
        for r in responses.rows
        if r["output"] and sampled(f"t{r['ticket']}", JUDGE_RATE, "judge") and r["id"] not in judged.ids
    ]

    def one(r: dict) -> None:
        prompt = JUDGE_PROMPT.format(ticket=r["input"], answer=r["output"])
        res = call([{"role": "user", "content": prompt}], "smart", 1000)
        verdict = res["output"].strip().upper()
        judged.add(
            {
                "id": r["id"],
                "run": r["run"],
                "ticket": r["ticket"],
                "verdict": verdict[:3],
                "yes": verdict.startswith("YES"),
                "cost_usd": res["cost_usd"],
                "error": res["error"],
            }
        )
        spend.add(res["cost_usd"])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    print(f"judge: {len(todo)} new verdicts, spend so far ${spend.total:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs", nargs="+", default=["v1", "v2", "v3", "smart", "judge"], choices=[*RUNS, "judge", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-spend", type=float, default=3.0, help="stop when this run's calls cost more (USD)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    responses, judged, spend = Store(out / "responses.jsonl"), Store(out / "judge.jsonl"), Spend(args.max_spend)
    for name in args.runs:
        if name == "test":
            r = call(
                [{"role": "system", "content": PROMPTS["v1"]}, {"role": "user", "content": make_ticket(0)["text"]}],
                "fast",
            )
            print(json.dumps(r))
            spend.add(r["cost_usd"])
        elif name == "judge":
            run_judge(responses, judged, spend, args.workers)
        else:
            run_responses(name, responses, spend, args.workers)


if __name__ == "__main__":
    main()
