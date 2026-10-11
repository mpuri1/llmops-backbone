"""Offline eval gate: score stored model outputs, diff a candidate against a baseline, fail on regressions.

No model is called. Inputs are JSONL files:

* cases:   {"id", "must_include": [...], "must_not_include": [...], "json": bool, "max_words": int}
* outputs: {"id", "output"}

Gate rule (written down so a failure is never a judgement call):

1. Metric rule: a metric fails when ``baseline - candidate > tolerance`` (absolute, default 0.05).
2. Case rule: the gate fails when more than ``max_case_regressions`` (default 0) cases passed in the
   baseline and fail in the candidate, even if the metrics net out.
3. A case with no candidate output counts as failing. Improvements never fail the gate.

Exit codes of the CLI: 0 pass, 1 gate failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "<!-- llmops-eval-gate -->"
DEFAULT_TOLERANCE = 0.05
DEFAULT_MAX_CASE_REGRESSIONS = 0
METRICS = ("pass_rate", "include_rate", "exclude_rate", "format_rate")


_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.S)


def loads_tolerant(output: str):
    """json.loads of a model response. A single code fence around the whole JSON is allowed, since models often add
    one; any other text around it is not. Raises json.JSONDecodeError when the response is not JSON."""
    text = output.strip()
    m = _FENCE.match(text)
    return json.loads(m.group(1).strip() if m else text)


class EvalInputError(ValueError):
    """A cases or outputs file is missing, malformed or inconsistent."""


@dataclass
class Scored:
    metrics: dict[str, float]
    passed: dict[str, bool]  # case id -> passed every check
    missing: list[str] = field(default_factory=list)  # case ids with no output


@dataclass
class MetricRow:
    name: str
    baseline: float
    candidate: float
    delta: float
    failed: bool


@dataclass
class Diff:
    rows: list[MetricRow]
    regressions: list[str]  # passed in baseline, fail in candidate
    fixed: list[str]  # failed in baseline, pass in candidate
    missing: list[str]  # cases the candidate has no output for
    tolerance: float
    max_case_regressions: int

    @property
    def failed(self) -> bool:
        return any(r.failed for r in self.rows) or len(self.regressions) > self.max_case_regressions


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as e:
        raise EvalInputError(f"cannot read {path}: {e}") from e
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvalInputError(f"{path} line {n}: invalid JSON ({e.msg})") from e
        if not isinstance(row, dict) or "id" not in row:
            raise EvalInputError(f"{path} line {n}: expected an object with an 'id'")
        rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise EvalInputError(f"{path}: duplicate ids")
    return rows


def _check_case(case: dict, output: str) -> dict[str, bool]:
    """Return {check_group: ok} for the groups this case defines."""
    text = output.lower()
    checks: dict[str, bool] = {}
    if case.get("must_include"):
        checks["include_rate"] = all(s.lower() in text for s in case["must_include"])
    if case.get("must_not_include"):
        checks["exclude_rate"] = not any(s.lower() in text for s in case["must_not_include"])
    fmt = []
    if case.get("json"):
        try:
            loads_tolerant(output)
            fmt.append(True)
        except json.JSONDecodeError:
            fmt.append(False)
    if case.get("max_words") is not None:
        fmt.append(len(output.split()) <= case["max_words"])
    if fmt:
        checks["format_rate"] = all(fmt)
    return checks


def score(cases: list[dict], outputs: list[dict]) -> Scored:
    if not cases:
        raise EvalInputError("no cases")
    by_id = {o["id"]: str(o.get("output", "")) for o in outputs}
    groups: dict[str, list[bool]] = {m: [] for m in METRICS}
    passed: dict[str, bool] = {}
    missing = []
    for case in cases:
        if case["id"] not in by_id:
            missing.append(case["id"])
        checks = _check_case(case, by_id.get(case["id"], ""))
        if case["id"] in missing:  # no output at all: every defined check fails
            checks = dict.fromkeys(checks, False)
        for name, ok in checks.items():
            groups[name].append(ok)
        passed[case["id"]] = all(checks.values()) and case["id"] not in missing
    metrics = {"pass_rate": sum(passed.values()) / len(passed)}
    for name in METRICS[1:]:
        if groups[name]:
            metrics[name] = sum(groups[name]) / len(groups[name])
    return Scored(metrics=metrics, passed=passed, missing=missing)


def diff(
    baseline: Scored,
    candidate: Scored,
    tolerance: float = DEFAULT_TOLERANCE,
    max_case_regressions: int = DEFAULT_MAX_CASE_REGRESSIONS,
) -> Diff:
    rows = []
    for name in METRICS:
        if name not in baseline.metrics:
            continue
        b = baseline.metrics[name]
        c = candidate.metrics.get(name, 0.0)
        rows.append(MetricRow(name, b, c, c - b, (b - c) > tolerance + 1e-9))
    regressions = sorted(i for i, ok in baseline.passed.items() if ok and not candidate.passed.get(i, False))
    fixed = sorted(i for i, ok in baseline.passed.items() if not ok and candidate.passed.get(i, False))
    return Diff(rows, regressions, fixed, sorted(candidate.missing), tolerance, max_case_regressions)


def render_markdown(d: Diff) -> str:
    verdict = "FAILED" if d.failed else "passed"
    lines = [
        MARKER,
        f"### Eval gate {verdict}",
        "",
        "| Metric | Baseline | Candidate | Delta | Status |",
        "|---|---|---|---|---|",
    ]
    for r in d.rows:
        status = "fail" if r.failed else ("improved" if r.delta > 1e-9 else "ok")
        lines.append(f"| {r.name} | {r.baseline:.3f} | {r.candidate:.3f} | {r.delta:+.3f} | {status} |")
    lines += [
        "",
        f"Rule: a metric fails if it drops by more than {d.tolerance:.3f}; "
        f"the gate also fails if more than {d.max_case_regressions} previously passing case(s) now fail.",
    ]
    if d.regressions:
        lines.append(f"- Newly failing cases ({len(d.regressions)}): {', '.join(f'`{i}`' for i in d.regressions)}")
    if d.fixed:
        lines.append(f"- Newly passing cases ({len(d.fixed)}): {', '.join(f'`{i}`' for i in d.fixed)}")
    if d.missing:
        lines.append(f"- Cases with no candidate output ({len(d.missing)}): {', '.join(f'`{i}`' for i in d.missing)}")
    lines.append("")
    lines.append("Offline comparison of stored outputs; no model was called.")
    return "\n".join(lines) + "\n"


def run(
    cases_path: str | Path,
    baseline_path: str | Path,
    candidate_path: str | Path,
    tolerance: float = DEFAULT_TOLERANCE,
    max_case_regressions: int = DEFAULT_MAX_CASE_REGRESSIONS,
) -> Diff:
    cases = read_jsonl(cases_path)
    return diff(
        score(cases, read_jsonl(baseline_path)),
        score(cases, read_jsonl(candidate_path)),
        tolerance,
        max_case_regressions,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m llmops_kit.evalgate", description=__doc__.split("\n")[0])
    p.add_argument("--cases", required=True)
    p.add_argument("--baseline", required=True, help="stored outputs of the current prompt")
    p.add_argument("--candidate", required=True, help="stored outputs of the prompt in the PR")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--max-case-regressions", type=int, default=DEFAULT_MAX_CASE_REGRESSIONS)
    p.add_argument("--markdown", help="write the PR comment body to this file")
    args = p.parse_args(argv)
    try:
        d = run(args.cases, args.baseline, args.candidate, args.tolerance, args.max_case_regressions)
    except EvalInputError as e:
        print(f"eval gate: input error: {e}", file=sys.stderr)
        return 2
    md = render_markdown(d)
    if args.markdown:
        Path(args.markdown).write_text(md)
    print(md)
    return 1 if d.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
