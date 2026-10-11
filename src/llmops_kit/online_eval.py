"""Online evaluation: score live responses without a reference, alert on drift, grow a golden set, canary a change.

The offline gate (evalgate.py) compares stored outputs against fixed cases before a merge. This module is the
production half. Nothing here calls a model; it works on the records your calls already produce.

* ``check_response``: reference-free checks (valid JSON, required keys, allowed values, length, banned strings).
* ``sampled``: a deterministic sampler by request id, so the same requests are judged on every replay.
* ``RateMonitor``: alerts when the upper end of the 95% Wilson interval of a pass rate over the last ``window``
  responses falls below ``baseline - tolerance``. ``MeanMonitor`` does the same for a cost or latency limit.
* ``FailureQueue`` and ``promote``: flagged responses wait for a reviewer; accepted ones become cases in the
  offline gate's format, so a production failure becomes a regression test.
* ``CanaryRouter`` and ``CanaryController``: send a fraction of requests to a candidate alias and roll it back
  to zero when the candidate's monitor alerts.

CLI:
    python -m llmops_kit.online_eval monitor --records log.jsonl --spec spec.json
    python -m llmops_kit.online_eval promote --queue queue.jsonl --golden cases.jsonl --decisions decisions.json

A record is {"id", "alias", "input", "output", "latency_s", "cost_usd"}. Exit codes: 0 no alert, 1 alert, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from llmops_kit.evalgate import EvalInputError, _check_case, loads_tolerant, read_jsonl

Z95 = 1.96


# ---------------------------------------------------------------- checks


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def parse_json(output: str):
    """The parsed JSON value, or None when the whole response is not JSON. A single code fence is tolerated."""
    try:
        return loads_tolerant(output)
    except json.JSONDecodeError:
        return None


def check_response(spec: dict, output: str) -> dict[str, bool]:
    """Reference-free checks for one response. Spec keys, all optional:

    ``json`` (the whole response is JSON), ``required_keys``, ``allowed`` ({key: [values]}), ``max_words``,
    ``must_not_include``. Returns {group: ok}; a response passes when every group is ok.
    """
    checks: dict[str, bool] = {}
    shared = _check_case({k: spec[k] for k in ("max_words", "must_not_include") if k in spec}, output)
    if "format_rate" in shared:
        checks["length"] = shared["format_rate"]
    if "exclude_rate" in shared:
        checks["exclude"] = shared["exclude_rate"]
    wants_json = spec.get("json") or spec.get("required_keys") or spec.get("allowed")
    if wants_json:
        value = parse_json(output)
        checks["json"] = value is not None
        if spec.get("required_keys") or spec.get("allowed"):
            ok = isinstance(value, dict) and all(k in value for k in spec.get("required_keys", []))
            for key, values in (spec.get("allowed") or {}).items():
                ok = ok and isinstance(value, dict) and value.get(key) in values
            checks["schema"] = bool(ok)
    return checks


def passed(checks: dict[str, bool]) -> bool:
    return all(checks.values())


# ---------------------------------------------------------------- sampling and statistics


def sampled(request_id: str, rate: float, salt: str = "") -> bool:
    """True for about ``rate`` of request ids, the same ones on every replay."""
    h = int(hashlib.sha256(f"{salt}{request_id}".encode()).hexdigest()[:8], 16) / 2**32
    return h < rate


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(outcomes: list[bool]) -> float:
    return sum(outcomes) / len(outcomes)


# ---------------------------------------------------------------- monitors


@dataclass
class Alert:
    monitor: str
    index: int  # observation number (1-based) at which it fired
    value: float
    threshold: float
    detail: str = ""


class RateMonitor:
    """Alert when the 95% Wilson upper bound of the last ``window`` pass/fail outcomes is below
    ``baseline - tolerance``. One alert per episode: it fires again only after the bound has recovered.
    Nothing is evaluated before ``min_n`` outcomes (default: a full window)."""

    def __init__(
        self,
        baseline: float,
        window: int = 100,
        tolerance: float = 0.05,
        z: float = Z95,
        name: str = "pass_rate",
        min_n: int | None = None,
    ):
        self.baseline, self.window, self.tolerance, self.z, self.name = baseline, window, tolerance, z, name
        self.min_n = min_n or window
        self.threshold = baseline - tolerance
        self.buf: deque[bool] = deque(maxlen=window)
        self.n = 0
        self.latched = False

    def observe(self, ok: bool) -> Alert | None:
        self.n += 1
        self.buf.append(bool(ok))
        if len(self.buf) < self.min_n:
            return None
        _, hi = wilson(sum(self.buf), len(self.buf), self.z)
        if hi < self.threshold:
            if not self.latched:
                self.latched = True
                return Alert(self.name, self.n, hi, self.threshold, f"window pass rate {rate(list(self.buf)):.3f}")
        else:
            self.latched = False
        return None


class MeanMonitor:
    """Alert when the mean of the last ``window`` values (cost per request, latency) exceeds ``baseline * factor``."""

    def __init__(self, baseline: float, window: int = 50, factor: float = 1.5, name: str = "mean"):
        self.baseline, self.window, self.factor, self.name = baseline, window, factor, name
        self.threshold = baseline * factor
        self.buf: deque[float] = deque(maxlen=window)
        self.n = 0
        self.latched = False

    def observe(self, value: float) -> Alert | None:
        self.n += 1
        self.buf.append(float(value))
        if len(self.buf) < self.window:
            return None
        mean = sum(self.buf) / len(self.buf)
        if mean > self.threshold:
            if not self.latched:
                self.latched = True
                return Alert(self.name, self.n, mean, self.threshold, f"baseline {self.baseline:.6g}")
        else:
            self.latched = False
        return None


# ---------------------------------------------------------------- events


def log_event(path: str | Path | None, kind: str, **fields) -> dict:
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "kind": kind, **fields}
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
    return event


# ---------------------------------------------------------------- failure queue and golden set


class FailureQueue:
    """Flagged responses waiting for a reviewer (JSONL, one per input; a repeat of the same input is ignored)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        return read_jsonl(self.path) if self.path.exists() else []

    def flag(self, record: dict, reason: str) -> bool:
        h = input_hash(record["input"])
        if any(r["input_hash"] == h for r in self.rows()):
            return False
        row = {
            "id": record["id"],
            "input_hash": h,
            "input": record["input"],
            "output": record.get("output", ""),
            "alias": record.get("alias"),
            "reason": reason,
            "status": "pending",
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return True


def promote(
    queue_path: str | Path, golden_path: str | Path, decisions: dict[str, dict | None], date: str = ""
) -> list[dict]:
    """Apply a reviewer's decisions. ``decisions`` maps a queue id to the case checks to enforce
    ({"must_include": [...], "json": true, ...}) or to None to reject. Accepted cases are appended to the golden
    file in the offline gate's format; an input already in the golden file is not added twice. Returns the
    cases added."""
    queue = FailureQueue(queue_path)
    rows = queue.rows()
    unknown = sorted(set(decisions) - {r["id"] for r in rows})
    if unknown:
        raise EvalInputError(f"decisions for ids not in the queue: {', '.join(unknown)}")
    golden = Path(golden_path)
    have = {c["id"] for c in read_jsonl(golden)} if golden.exists() else set()
    added = []
    for row in rows:
        if row["id"] not in decisions or row["status"] != "pending":
            continue
        spec = decisions[row["id"]]
        if spec is None:
            row["status"] = "rejected"
            continue
        case_id = f"g-{row['input_hash']}"
        row["status"] = "promoted"
        if case_id in have:
            continue
        case = {"id": case_id, "input": row["input"], **spec, "source": row["id"]}
        if date:
            case["added"] = date
        added.append(case)
        have.add(case_id)
    if added:
        with open(golden, "a") as f:
            for case in added:
                f.write(json.dumps(case) + "\n")
    queue.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return added


# ---------------------------------------------------------------- canary


class CanaryRouter:
    """Deterministic traffic split: about ``fraction`` of request ids go to the candidate alias."""

    def __init__(self, stable: str, candidate: str, fraction: float = 0.1, salt: str = "canary"):
        self.stable, self.candidate, self.fraction, self.salt = stable, candidate, fraction, salt
        self.rolled_back = False

    def choose(self, request_id: str) -> str:
        return self.candidate if sampled(request_id, self.fraction, self.salt) else self.stable

    def rollback(self) -> None:
        self.fraction = 0.0
        self.rolled_back = True


class CanaryController:
    """Watch the candidate's pass rate against the stable baseline; roll back when its monitor alerts."""

    def __init__(
        self,
        router: CanaryRouter,
        baseline: float,
        window: int = 100,
        tolerance: float = 0.05,
        events: str | Path | None = None,
        min_n: int = 20,
    ):
        self.router = router
        # the candidate's monitor starts empty, so it is allowed to judge after ``min_n`` candidate responses
        self.monitor = RateMonitor(baseline, window, tolerance, name=f"canary:{router.candidate}", min_n=min_n)
        self.events = events

    def observe(self, alias: str, ok: bool) -> Alert | None:
        if alias != self.router.candidate or self.router.rolled_back:
            return None
        alert = self.monitor.observe(ok)
        if alert:
            self.router.rollback()
            log_event(self.events, "canary_rollback", candidate=self.router.candidate, **asdict(alert))
        return alert


# ---------------------------------------------------------------- CLI


def monitor_records(
    records: list[dict], spec: dict, baseline_n: int = 200, window: int = 100, tolerance: float = 0.05
) -> list[Alert]:
    """Run the checks and both monitors over a production log. The first ``baseline_n`` records set the baselines."""
    if len(records) <= baseline_n:
        raise EvalInputError(f"need more than {baseline_n} records to set a baseline, got {len(records)}")
    outcomes = [passed(check_response(spec, r.get("output", ""))) for r in records]
    base = rate(outcomes[:baseline_n])
    rm = RateMonitor(base, window, tolerance)
    alerts = []
    costs = [r.get("cost_usd") for r in records]
    cm = None
    if all(c is not None for c in costs[:baseline_n]) and sum(costs[:baseline_n]) > 0:
        cm = MeanMonitor(sum(costs[:baseline_n]) / baseline_n, name="cost_usd")
    for i, ok in enumerate(outcomes[baseline_n:], baseline_n):
        for a in (rm.observe(ok), cm.observe(costs[i]) if cm and costs[i] is not None else None):
            if a:
                alerts.append(a)
    return alerts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m llmops_kit.online_eval", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor", help="run the checks and monitors over a production log")
    m.add_argument("--records", required=True)
    m.add_argument("--spec", required=True, help="JSON file with the response checks")
    m.add_argument("--baseline-n", type=int, default=200)
    m.add_argument("--window", type=int, default=100)
    m.add_argument("--tolerance", type=float, default=0.05)
    g = sub.add_parser("promote", help="apply review decisions: queue -> golden cases")
    g.add_argument("--queue", required=True)
    g.add_argument("--golden", required=True)
    g.add_argument("--decisions", required=True, help='JSON file: {"<queue id>": {"must_include": [...]} or null}')
    g.add_argument("--date", default="")
    args = p.parse_args(argv)
    try:
        if args.cmd == "monitor":
            spec = json.loads(Path(args.spec).read_text())
            alerts = monitor_records(read_jsonl(args.records), spec, args.baseline_n, args.window, args.tolerance)
            for a in alerts:
                print(json.dumps(asdict(a)))
            print(f"{len(alerts)} alert(s)", file=sys.stderr)
            return 1 if alerts else 0
        decisions = json.loads(Path(args.decisions).read_text())
        added = promote(args.queue, args.golden, decisions, args.date)
        print(f"{len(added)} case(s) added to {args.golden}")
        return 0
    except (EvalInputError, OSError, json.JSONDecodeError) as e:
        print(f"online eval: input error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
