"""Characterise the drift monitor on simulated pass/fail streams (no model, no cost).

For each true pass rate after a change, over many seeded streams: how many requests after the change does
``RateMonitor`` alert (the detection delay), in what share of streams does it alert at all, and how often does
it alert on a clean stream (false alarm episodes per 1,000 requests)?

    uv run python examples/online_evals/simulate.py --seeds 2000 --out results.json

The monitor's alert rule is evaluated through a precomputed count limit (the Wilson upper bound grows with the
count), which gives the same alerts as ``RateMonitor``; ``--check`` compares the two on a few streams.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

from llmops_kit.online_eval import RateMonitor, rate, wilson


def count_limit(window: int, threshold: float, z: float) -> int:
    """Largest count k in a window with Wilson upper bound below the threshold (-1 if none)."""
    limit = -1
    for k in range(window + 1):
        if wilson(k, window, z)[1] < threshold:
            limit = k
    return limit


def episodes(outcomes: list[bool], baseline: float, window: int, tolerance: float, z: float, start: int) -> list[int]:
    """Request numbers (1-based, in the whole stream) at which an alert episode begins, from ``start`` on."""
    limit = count_limit(window, baseline - tolerance, z)
    out, count, latched = [], 0, False
    obs = outcomes[start:]
    for i, ok in enumerate(obs):
        count += ok
        if i >= window:
            count -= obs[i - window]
        if i + 1 < window:
            continue
        if count <= limit:
            if not latched:
                latched = True
                out.append(start + i + 1)
        else:
            latched = False
    return out


def stream(rng: random.Random, p_before: float, p_after: float, change_at: int, length: int) -> list[bool]:
    return [rng.random() < (p_before if i < change_at else p_after) for i in range(length)]


def check_equivalence(seed: int = 0) -> None:
    rng = random.Random(seed)
    for p_after in (0.95, 0.85, 0.55):
        s = stream(rng, 0.95, p_after, 400, 1400)
        base = rate(s[:200])
        m = RateMonitor(base, 100, 0.05)
        slow = [200 + m.n for o in s[200:] if m.observe(o)]
        slow = [a for a in slow]
        # RateMonitor.n counts observations after the reference period
        fast = episodes(s, base, 100, 0.05, 1.96, 200)
        assert slow == fast, (slow, fast)


def run(
    seeds: int, windows: list[int], drops: list[float], ref_n=200, change_at=400, length=1400, tolerance=0.05, z=1.96
):
    results = {}
    for window in windows:
        for p_after in drops:
            delays, detected, false_alarms = [], 0, 0
            for seed in range(seeds):
                rng = random.Random(f"{window}-{p_after}-{seed}")
                s = stream(rng, 0.95, p_after, change_at, length)
                base = rate(s[:ref_n])
                ep = episodes(s, base, window, tolerance, z, ref_n)
                false_alarms += sum(1 for e in ep if e <= change_at)
                after = [e for e in ep if e > change_at]
                if after:
                    detected += 1
                    delays.append(after[0] - change_at)
            monitored_before = change_at - ref_n if p_after != 0.95 else length - ref_n
            key = f"w{window}_after{p_after}"
            results[key] = {
                "window": window,
                "true_rate_after": p_after,
                "streams": seeds,
                "detected_share": detected / seeds,
                "delay_median": statistics.median(delays) if delays else None,
                "delay_p90": sorted(delays)[int(0.9 * len(delays)) - 1] if len(delays) >= 10 else None,
                "false_alarm_episodes_per_1000_requests": 1000 * false_alarms / (seeds * monitored_before),
            }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--windows", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument(
        "--drops",
        type=float,
        nargs="+",
        default=[0.95, 0.90, 0.85, 0.75, 0.55],
        help="true pass rate after the change; 0.95 = no change",
    )
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--check", action="store_true", help="check the fast path against RateMonitor and exit")
    args = ap.parse_args()
    if args.check:
        for seed in range(20):
            check_equivalence(seed)
        print("fast path matches RateMonitor on 60 streams")
        return
    results = run(args.seeds, args.windows, args.drops)
    for key, r in results.items():
        print(
            key,
            json.dumps(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()
                    if k not in ("window", "streams")
                }
            ),
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
