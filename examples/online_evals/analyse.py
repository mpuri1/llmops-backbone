"""Replay the stored E2 responses through the online-eval pieces and report what each one saw.

Reads <out-dir>/responses.jsonl and judge.jsonl (from collect.py) and writes <out-dir>/analysis.json. No model is
called. Parts:

* monitors: reference-free pass rate, cost per request and the sampled judge, on the stable-then-regression stream
  of each run (the change is at request 300), and on the clean control.
* judge against the labels (Cohen's kappa).
* loop: judge-flagged failures -> reviewer -> golden cases -> the offline gate, before and after.
* canary: a replay of the format regression at 10% traffic against a full rollout.

The labels are used to measure the monitors and to stand in for the reviewer; no monitor reads them.

    uv run python examples/online_evals/analyse.py --out-dir results/e2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tickets import SPEC, make_ticket

from llmops_kit import evalgate
from llmops_kit import online_eval as oe

CHANGE_AT = 300  # requests; the regression starts at request 301
BASELINE_N = 200
RATE_WINDOW = 100
JUDGE_WINDOW = 30
COST_WINDOW = 50
COST_FACTOR = 1.5
RUNS = {"control": None, "R1_format": "v2", "R2_silent": "v3", "R3a_truncated": "smart", "R3b_cost": "smart2"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def truth(row: dict) -> dict:
    label = make_ticket(row["ticket"])["label"]
    value = oe.parse_json(row["output"])
    pred = value if isinstance(value, dict) else {}
    cat, pri, ref = (pred.get(k) == label[k] for k in ("category", "priority", "refund_requested"))
    return {"cat": cat, "pri": pri, "ref": ref, "cp": cat and pri, "all": cat and pri and ref}


def stream(by_id: dict, run: str | None) -> list[dict]:
    """600 responses in ticket order: the stable run, with tickets 300-599 swapped for the regression run."""
    if run is None:
        return [by_id[f"v1-{i}"] for i in range(600)]
    return [by_id[f"v1-{i}"] for i in range(300)] + [by_id[f"{run}-{i}"] for i in range(300, 600)]


def first_after(requests: list[int], start: int = 0) -> int | None:
    later = [r for r in requests if r > start]
    return later[0] if later else None


def monitors(rows: list[dict], judged: dict[str, bool]) -> dict:
    outcomes = [oe.passed(oe.check_response(SPEC, r["output"])) for r in rows]
    base = oe.rate(outcomes[:BASELINE_N])
    rm = oe.RateMonitor(base, RATE_WINDOW, 0.05)
    rate_alerts = [i for i, o in enumerate(outcomes[BASELINE_N:], BASELINE_N + 1) if rm.observe(o)]
    costs = [r["cost_usd"] or 0.0 for r in rows]
    cm = oe.MeanMonitor(statistics.mean(costs[:BASELINE_N]), COST_WINDOW, COST_FACTOR, "cost_usd")
    cost_alerts = [i for i, c in enumerate(costs[BASELINE_N:], BASELINE_N + 1) if cm.observe(c)]
    jrows = [(r["ticket"], judged[r["id"]]) for r in rows if r["id"] in judged]
    stable = [y for t, y in jrows if t < CHANGE_AT]
    judge_alerts = []
    if len(stable) >= 10:
        jm = oe.RateMonitor(oe.rate(stable), JUDGE_WINDOW, 0.05, name="judge_yes")
        judge_alerts = [t + 1 for t, y in jrows if t >= CHANGE_AT and jm.observe(y)]
    first = lambda xs: first_after(xs, 0)  # noqa: E731
    return {
        "reference_free_pass_rate_before": oe.rate(outcomes[:CHANGE_AT]),
        "reference_free_pass_rate_after": oe.rate(outcomes[CHANGE_AT:]),
        "rate_alert_requests": rate_alerts,
        "rate_alert_delay": None if not first(rate_alerts) else first(rate_alerts) - CHANGE_AT,
        "cost_alert_delay": None if not first(cost_alerts) else first(cost_alerts) - CHANGE_AT,
        "judge_alert_delay": None if not first(judge_alerts) else first(judge_alerts) - CHANGE_AT,
        "judged_before": len(stable),
        "judged_after": len(jrows) - len(stable),
        "judge_yes_before": oe.rate(stable) if stable else None,
        "judge_yes_after": oe.rate([y for t, y in jrows if t >= CHANGE_AT]) if len(jrows) > len(stable) else None,
        "mean_cost_before": statistics.mean(costs[:CHANGE_AT]),
        "mean_cost_after": statistics.mean(costs[CHANGE_AT:]),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def accuracy(rows: list[dict]) -> dict:
    t = [truth(r) for r in rows]
    return {k: sum(x[k] for x in t) / len(t) for k in ("cat", "pri", "ref", "all")}


def judge_vs_truth(rows_all: list[dict], judged: dict[str, bool]) -> dict:
    pairs = [(judged[r["id"]], truth(r)["cp"]) for r in rows_all if r["id"] in judged]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "agreement": sum(x == y for x, y in pairs) / len(pairs),
        "kappa": kappa(a, b),
        "judge_yes": sum(a) / len(a),
        "truth_correct": sum(b) / len(b),
    }


def gate_cases(by_id: dict) -> list[dict]:
    singles = [i for i in range(300, 400) if make_ticket(i)["issues"] == 1][:40]
    return [
        {
            "id": f"o-{i}",
            "input": make_ticket(i)["text"],
            "must_include": [make_ticket(i)["label"]["category"]],
            "json": True,
        }
        for i in singles
    ]


def outputs(by_id: dict, run: str, cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        i = int(c["id"].split("-")[1]) if c["id"].startswith("o-") else int(c["source"].split("-")[1])
        out.append({"id": c["id"], "output": by_id[f"{run}-{i}"]["output"]})
    return out


def gate(cases: list[dict], by_id: dict, run: str) -> dict:
    d = evalgate.diff(
        evalgate.score(cases, outputs(by_id, "v1", cases)), evalgate.score(cases, outputs(by_id, run, cases))
    )
    return {
        "cases": len(cases),
        "failed": d.failed,
        "newly_failing": len(d.regressions),
        "metrics": {r.name: [round(r.baseline, 3), round(r.candidate, 3)] for r in d.rows},
    }


def loop(by_id: dict, judge_rows: list[dict], work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    for f in ("queue.jsonl", "golden.jsonl"):
        (work / f).unlink(missing_ok=True)
    original = gate_cases(by_id)
    judged = {j["id"]: j["yes"] for j in judge_rows}
    queue = oe.FailureQueue(work / "queue.jsonl")
    flagged = []
    for i in range(300, 450):  # the first half of the regression period is what "production" has seen so far
        r = by_id[f"v3-{i}"]
        if (
            r["id"] in judged
            and not judged[r["id"]]
            and queue.flag({**r, "id": r["id"], "input": r["input"]}, "judge said NO")
        ):
            flagged.append(r)
    decisions, rejected = {}, 0
    for r in flagged:  # the reviewer is the generator's label: accept only a response that is truly wrong
        label = make_ticket(r["ticket"])["label"]
        if truth(r)["cp"]:
            decisions[r["id"]] = None
            rejected += 1
        else:
            decisions[r["id"]] = {"must_include": [label["category"]], "json": True}
    added = oe.promote(work / "queue.jsonl", work / "golden.jsonl", decisions, date="2026-10-10")
    golden = evalgate.read_jsonl(work / "golden.jsonl") if (work / "golden.jsonl").exists() else []
    return {
        "original_cases": len(original),
        "judge_flagged": len(flagged),
        "reviewer_rejected": rejected,
        "promoted": len(added),
        "gate_on_original_cases": {run: gate(original, by_id, run) for run in ("v2", "v3", "smart", "smart2")},
        "gate_after_promotion": gate(original + golden, by_id, "v3") if golden else None,
    }


def canary(by_id: dict, window: int, min_n: int, total=3000, start=500, fraction=0.1) -> dict:
    stable = [by_id[f"v1-{i}"] for i in range(300, 600)]
    cand = [by_id[f"v2-{i}"] for i in range(300, 600)]
    bad = lambda r: not oe.passed(oe.check_response(SPEC, r["output"]))  # noqa: E731
    base = oe.rate([not bad(r) for r in stable[:BASELINE_N]])
    # canary
    router = oe.CanaryRouter("stable", "candidate", fraction)
    ctl = oe.CanaryController(router, base, window=window, min_n=min_n)
    bad_served, rolled_at = 0, None
    for k in range(total):
        row_i = k % 300
        alias = router.choose(f"req-{k}") if k >= start else "stable"
        r = cand[row_i] if alias == "candidate" else stable[row_i]
        if alias == "candidate":
            bad_served += bad(r)
            if ctl.observe("candidate", not bad(r)) and rolled_at is None:
                rolled_at = k + 1 - start
    # full rollout watched by the same rule on all traffic
    mon = oe.RateMonitor(base, window, 0.05)
    full_bad, full_rolled = 0, None
    for k in range(total):
        row_i = k % 300
        if k >= start and full_rolled is None:
            r = cand[row_i]
            full_bad += bad(r)
            if mon.observe(not bad(r)):
                full_rolled = k + 1 - start
        else:
            r = stable[row_i]
            if full_rolled is None:
                mon.observe(not bad(r))
    return {
        "window": window,
        "candidate_min_n": min_n,
        "canary_rollback_after_requests": rolled_at,
        "canary_bad_served": bad_served,
        "full_rollout_rollback_after_requests": full_rolled,
        "full_rollout_bad_served": full_bad,
        "canary_share_of_traffic_exposed": fraction,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    responses, judge_rows = load(out / "responses.jsonl"), load(out / "judge.jsonl")
    by_id = {r["id"]: r for r in responses}
    judged = {j["id"]: j["yes"] for j in judge_rows if not j.get("error") and j["verdict"] in ("YES", "NO")}
    result = {
        "responses": len(responses),
        "judged": len(judge_rows),
        "judge_invalid_verdicts": len(judge_rows) - len(judged),
        "runs": {},
    }
    for name, run in RUNS.items():
        rows = stream(by_id, run)
        result["runs"][name] = {
            "monitors": monitors(rows, judged),
            "accuracy_before": accuracy(rows[:CHANGE_AT]),
            "accuracy_after": accuracy(rows[CHANGE_AT:]),
        }
    result["judge_vs_truth"] = judge_vs_truth([r for r in responses if r["output"]], judged)
    result["loop"] = loop(by_id, judge_rows, out / "loop")
    result["canary"] = [canary(by_id, 100, 100), canary(by_id, 100, 20), canary(by_id, 25, 25)]
    result["spend_usd"] = sum((r.get("cost_usd") or 0) for r in responses + judge_rows)
    (out / "analysis.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
