import json
import random
import subprocess
import sys

import pytest

from llmops_kit import evalgate
from llmops_kit import online_eval as oe

SPEC = {
    "json": True,
    "required_keys": ["category", "priority"],
    "allowed": {"category": ["billing", "bug"], "priority": ["low", "high"]},
    "max_words": 30,
    "must_not_include": ["as an ai"],
}


def ok_output(category="billing", priority="low"):
    return json.dumps({"category": category, "priority": priority})


def test_check_response_passes_a_valid_answer_and_tolerates_one_fence():
    assert oe.passed(oe.check_response(SPEC, ok_output()))
    assert oe.passed(oe.check_response(SPEC, "```json\n" + ok_output() + "\n```"))


@pytest.mark.parametrize(
    "output, failing",
    [
        ("Sure! Here you go: " + ok_output(), "json"),
        (json.dumps({"category": "billing"}), "schema"),
        (ok_output(category="sales"), "schema"),
        (json.dumps({"category": "bug", "priority": "high", "note": "word " * 40}), "length"),
        (json.dumps({"category": "bug", "priority": "high", "note": "As an AI"}), "exclude"),
        ("[1, 2]", "schema"),
    ],
)
def test_check_response_names_the_failing_group(output, failing):
    checks = oe.check_response(SPEC, output)
    assert checks[failing] is False
    assert not oe.passed(checks)


def test_check_response_without_a_spec_passes_everything():
    assert oe.check_response({}, "anything") == {}
    assert oe.passed({})


def test_wilson_interval_matches_known_values():
    lo, hi = oe.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert oe.wilson(0, 0) == (0.0, 1.0)
    lo, hi = oe.wilson(100, 100)
    assert hi == pytest.approx(1.0) and lo > 0.95


def test_sampler_is_deterministic_and_close_to_the_rate():
    ids = [f"req-{i}" for i in range(5000)]
    first = [oe.sampled(i, 0.2) for i in ids]
    assert first == [oe.sampled(i, 0.2) for i in ids]
    assert 0.17 < sum(first) / len(first) < 0.23
    assert not any(oe.sampled(i, 0.0) for i in ids) and all(oe.sampled(i, 1.0) for i in ids)
    assert first != [oe.sampled(i, 0.2, salt="other") for i in ids]


def feed(monitor, outcomes):
    return [a for a in (monitor.observe(o) for o in outcomes) if a]


def test_rate_monitor_stays_quiet_on_a_clean_stream_and_before_the_window_fills():
    m = oe.RateMonitor(0.95, window=100)
    assert feed(m, [False] * 99) == []  # window not full yet
    rng = random.Random(1)
    m = oe.RateMonitor(0.95, window=100)
    assert feed(m, [rng.random() < 0.95 for _ in range(1000)]) == []


def test_rate_monitor_alerts_once_per_episode_and_rearms_after_recovery():
    m = oe.RateMonitor(0.95, window=50)
    alerts = feed(m, [True] * 60 + [False] * 80)
    assert len(alerts) == 1 and alerts[0].index > 60 and alerts[0].value < alerts[0].threshold
    assert feed(m, [True] * 100) == []  # recovers, latch released
    assert len(feed(m, [False] * 80)) == 1  # second episode alerts again


def test_a_drop_equal_to_the_tolerance_is_not_an_alert():
    m = oe.RateMonitor(0.95, window=100, tolerance=0.05)
    rng = random.Random(2)
    assert feed(m, [rng.random() < 0.90 for _ in range(300)]) == []


def test_mean_monitor_alerts_when_the_window_mean_passes_the_limit():
    m = oe.MeanMonitor(0.001, window=10, factor=2.0, name="cost_usd")
    assert feed(m, [0.001] * 30) == []
    alerts = feed(m, [0.01] * 30)
    assert len(alerts) == 1 and alerts[0].monitor == "cost_usd" and alerts[0].value > alerts[0].threshold


def record(i, output="x"):
    return {"id": f"r{i}", "input": f"ticket {i}", "output": output, "alias": "fast"}


def test_failure_queue_ignores_a_repeated_input(tmp_path):
    q = oe.FailureQueue(tmp_path / "q.jsonl")
    assert q.flag(record(1), "schema") is True
    assert q.flag({**record(2), "input": "ticket 1"}, "schema") is False
    assert [r["id"] for r in q.rows()] == ["r1"]


def test_promote_turns_accepted_failures_into_cases_the_offline_gate_can_score(tmp_path):
    q, g = tmp_path / "q.jsonl", tmp_path / "golden.jsonl"
    queue = oe.FailureQueue(q)
    for i in (1, 2, 3):
        queue.flag(record(i, output="bad"), "schema")
    added = oe.promote(q, g, {"r1": {"must_include": ["billing"], "json": True}, "r2": None}, date="2026-10-10")
    assert [c["source"] for c in added] == ["r1"] and added[0]["added"] == "2026-10-10"
    status = {r["id"]: r["status"] for r in queue.rows()}
    assert status == {"r1": "promoted", "r2": "rejected", "r3": "pending"}
    cases = evalgate.read_jsonl(g)
    scored = evalgate.score(cases, [{"id": cases[0]["id"], "output": '{"category": "billing"}'}])
    assert scored.metrics["pass_rate"] == 1.0
    assert oe.promote(q, g, {"r1": {"must_include": ["billing"]}}) == []  # already decided: nothing added twice


def test_promote_does_not_add_an_input_that_is_already_in_the_golden_file(tmp_path):
    q, g = tmp_path / "q.jsonl", tmp_path / "golden.jsonl"
    oe.FailureQueue(q).flag(record(1), "schema")
    g.write_text(json.dumps({"id": f"g-{oe.input_hash('ticket 1')}", "input": "ticket 1"}) + "\n")
    assert oe.promote(q, g, {"r1": {"json": True}}) == []
    assert len(evalgate.read_jsonl(g)) == 1


def test_promote_rejects_ids_that_are_not_in_the_queue(tmp_path):
    oe.FailureQueue(tmp_path / "q.jsonl").flag(record(1), "schema")
    with pytest.raises(evalgate.EvalInputError, match="not in the queue"):
        oe.promote(tmp_path / "q.jsonl", tmp_path / "g.jsonl", {"nope": None})


def test_canary_router_splits_close_to_the_fraction_and_rollback_sends_everything_to_stable():
    router = oe.CanaryRouter("fast", "smart", fraction=0.1)
    picks = [router.choose(f"req-{i}") for i in range(5000)]
    assert 0.08 < picks.count("smart") / len(picks) < 0.12
    assert picks == [router.choose(f"req-{i}") for i in range(5000)]
    router.rollback()
    assert {router.choose(f"req-{i}") for i in range(500)} == {"fast"} and router.rolled_back


def test_canary_controller_rolls_back_on_a_bad_candidate_and_logs_it(tmp_path):
    events = tmp_path / "events.jsonl"
    router = oe.CanaryRouter("fast", "smart", fraction=0.1)
    ctl = oe.CanaryController(router, baseline=0.95, window=40, events=events)
    for _ in range(100):  # stable traffic never moves the candidate's monitor
        assert ctl.observe("fast", False) is None
    fired = [ctl.observe("smart", False) for _ in range(60)]
    assert any(fired) and router.fraction == 0.0 and router.rolled_back
    assert ctl.observe("smart", False) is None  # after rollback nothing more fires
    event = json.loads(events.read_text().splitlines()[0])
    assert event["kind"] == "canary_rollback" and event["candidate"] == "smart"


def log_rows(n_good, n_bad, cost_good=0.001, cost_bad=0.001):
    rows = [{"id": f"r{i}", "input": "t", "output": ok_output(), "cost_usd": cost_good} for i in range(n_good)]
    rows += [{"id": f"r{n_good + i}", "input": "t", "output": "oops", "cost_usd": cost_bad} for i in range(n_bad)]
    return rows


def test_monitor_records_finds_a_quality_drop_and_a_cost_jump():
    assert oe.monitor_records(log_rows(400, 0), SPEC) == []
    quality = oe.monitor_records(log_rows(300, 200), SPEC)
    assert [a.monitor for a in quality] == ["pass_rate"]
    cost = oe.monitor_records(
        log_rows(300, 0) + [{**r, "id": f"c{i}", "cost_usd": 0.01} for i, r in enumerate(log_rows(120, 0))], SPEC
    )
    assert [a.monitor for a in cost] == ["cost_usd"]
    with pytest.raises(evalgate.EvalInputError, match="need more than"):
        oe.monitor_records(log_rows(50, 0), SPEC)


def test_cli_exit_codes(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))

    def run(rows, *extra):
        path = tmp_path / "log.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "llmops_kit.online_eval",
                "monitor",
                "--records",
                str(path),
                "--spec",
                str(spec),
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    assert run(log_rows(400, 0)).returncode == 0
    bad = run(log_rows(300, 200))
    assert bad.returncode == 1 and json.loads(bad.stdout.splitlines()[0])["monitor"] == "pass_rate"
    assert run(log_rows(10, 0)).returncode == 2


def test_the_offline_gate_and_the_online_checks_agree_on_what_counts_as_json():
    fenced = '```json\n{"a": 1}\n```'
    prose = 'Here you go: {"a": 1}'
    for output, valid in ((fenced, True), ('{"a": 1}', True), (prose, False), ("not json", False)):
        gate = evalgate.score([{"id": "x", "json": True}], [{"id": "x", "output": output}]).passed["x"]
        online = oe.check_response({"json": True}, output)["json"]
        assert gate == online == valid


def test_rate_monitor_can_judge_before_its_window_is_full_when_min_n_is_set():
    full = oe.RateMonitor(1.0, window=100)
    early = oe.RateMonitor(1.0, window=100, min_n=10)
    assert feed(full, [False] * 30) == []
    alerts = feed(early, [False] * 30)
    assert len(alerts) == 1 and alerts[0].index == 10


def test_canary_controller_judges_after_min_n_candidate_responses():
    router = oe.CanaryRouter("fast", "smart", fraction=0.1)
    ctl = oe.CanaryController(router, baseline=1.0, window=100, min_n=15)
    fired = [ctl.observe("smart", False) for _ in range(30)]
    assert sum(1 for f in fired if f) == 1 and fired[14] is not None and router.rolled_back
