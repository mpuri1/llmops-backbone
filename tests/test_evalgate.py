import json
import subprocess
import sys
from pathlib import Path

import pytest

from llmops_kit import evalgate

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "eval_gate"


def cli(*extra):
    return subprocess.run(
        [sys.executable, "-m", "llmops_kit.evalgate", "--cases", str(FIX / "cases.jsonl"), *extra],
        capture_output=True,
        text=True,
    )


def test_each_check_is_scored_separately():
    cases = [
        {"id": "a", "must_include": ["Refund"], "json": True},
        {"id": "b", "must_not_include": ["secret"]},
        {"id": "c", "max_words": 2},
    ]
    outs = [
        {"id": "a", "output": '{"x": "refund"}'},  # case-insensitive match, valid JSON
        {"id": "b", "output": "a SECRET here"},
        {"id": "c", "output": "one two three"},
    ]
    s = evalgate.score(cases, outs)
    assert s.passed == {"a": True, "b": False, "c": False}
    assert s.metrics["include_rate"] == 1.0
    assert s.metrics["exclude_rate"] == 0.0
    assert s.metrics["format_rate"] == 0.5  # case a (json) ok, case c (words) not
    assert s.metrics["pass_rate"] == pytest.approx(1 / 3)


def test_missing_output_fails_the_case():
    s = evalgate.score([{"id": "a", "must_not_include": ["x"]}], [])
    assert s.passed == {"a": False}
    assert s.missing == ["a"]


def test_tolerance_boundary_is_inclusive():
    base = evalgate.Scored({"pass_rate": 1.0}, {"a": True})
    cand = evalgate.Scored({"pass_rate": 0.95}, {"a": True})
    assert not evalgate.diff(base, cand, tolerance=0.05).failed  # drop equals tolerance: ok
    assert evalgate.diff(base, cand, tolerance=0.04).failed


def test_case_regression_fails_even_when_metrics_net_out():
    base = evalgate.Scored({"pass_rate": 0.5}, {"a": True, "b": False})
    cand = evalgate.Scored({"pass_rate": 0.5}, {"a": False, "b": True})
    d = evalgate.diff(base, cand)
    assert d.regressions == ["a"] and d.fixed == ["b"]
    assert d.failed
    assert not evalgate.diff(base, cand, max_case_regressions=1).failed


def test_improvement_passes():
    base = evalgate.Scored({"pass_rate": 0.5}, {"a": True, "b": False})
    cand = evalgate.Scored({"pass_rate": 1.0}, {"a": True, "b": True})
    d = evalgate.diff(base, cand)
    assert not d.failed and d.rows[0].delta == 0.5


def test_bad_input_raises(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(evalgate.EvalInputError):
        evalgate.read_jsonl(bad)
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "a"}\n{"id": "a"}\n')
    with pytest.raises(evalgate.EvalInputError):
        evalgate.read_jsonl(dup)


def test_cli_passing_fixture_exits_zero_and_writes_comment(tmp_path):
    out = tmp_path / "comment.md"
    r = cli(
        "--baseline",
        str(FIX / "baseline.jsonl"),
        "--candidate",
        str(FIX / "candidate_pass.jsonl"),
        "--markdown",
        str(out),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    body = out.read_text()
    assert body.startswith(evalgate.MARKER) and "Eval gate passed" in body


def test_cli_failing_fixture_exits_one_and_lists_regressions(tmp_path):
    out = tmp_path / "comment.md"
    r = cli(
        "--baseline",
        str(FIX / "baseline.jsonl"),
        "--candidate",
        str(FIX / "candidate_fail.jsonl"),
        "--markdown",
        str(out),
    )
    assert r.returncode == 1
    body = out.read_text()
    assert "Eval gate FAILED" in body
    for case_id in ("t01", "t07", "t09", "t10"):
        assert f"`{case_id}`" in body


def test_cli_bad_input_exits_two():
    r = cli("--baseline", str(FIX / "nope.jsonl"), "--candidate", str(FIX / "baseline.jsonl"))
    assert r.returncode == 2


def test_repo_gate_files_pass_as_committed():
    # The live gate (evals/) must be green on main: candidate starts identical to baseline.
    d = evalgate.run(ROOT / "evals/cases.jsonl", ROOT / "evals/baseline.jsonl", ROOT / "evals/candidate.jsonl")
    assert not d.failed
    json.dumps([r.name for r in d.rows])
