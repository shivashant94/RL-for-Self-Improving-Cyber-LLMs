"""CLI script: run Review 2 fixture episodes and write a summary report.

Usage
-----
    python3 scripts/run_review2_episode.py [--n-episodes N] [--seed SEED]
                                            [--output-dir DIR] [--dry-run]

Outputs
-------
    <output-dir>/review2_episode_report.json  summary over all episodes
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defender_policy.episode_harness import FixtureEpisodeHarness  # noqa: E402
from defender_policy.rewards import RewardWeights  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Review 2 fixture episodes.")
    parser.add_argument("--n-episodes", type=int, default=15,
                        help="Number of episodes to run (default: 15, one per eval case).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"),
                        help="Directory for output report (default: reports/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without writing any files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    harness = FixtureEpisodeHarness(seed=args.seed)
    results  = harness.run_n_episodes(args.n_episodes)

    # ── Aggregate statistics ─────────────────────────────────────────────
    total  = len(results)
    attack_results  = [r for r in results if r.case_kind == "attack"]
    benign_results  = [r for r in results if r.case_kind == "benign"]

    avg_reward  = sum(r.total_reward for r in results) / total if total else 0.0
    gate_blocks = sum(1 for r in results if r.gate_blocked)
    pre_unsafe  = sum(1 for r in results if r.pre_gate_unsafe)

    # Capability retention: take the last episode's report (all are identical
    # because the fixture probe bank is static)
    retention = results[-1].retention_report if results else None

    # Per-family reward
    family_rewards: dict = {}
    for r in attack_results:
        fam = r.attack_family or "unknown"
        family_rewards.setdefault(fam, []).append(r.total_reward)
    per_family_avg = {fam: sum(v)/len(v) for fam, v in family_rewards.items()}

    # Build per-episode trace summary (sanitized — no raw untrusted content)
    episode_traces = []
    for r in results:
        step = r.steps[0]
        episode_traces.append({
            "episode_id":    r.episode_id,
            "case_id":       r.case_id,
            "case_kind":     r.case_kind,
            "attack_family": r.attack_family,
            "defender_status": step.action.answer_text[:80] if step.action.answer_text else "<tool_call>",
            "gate_blocked":  r.gate_blocked,
            "pre_gate_unsafe": r.pre_gate_unsafe,
            "total_reward":  r.total_reward,
            "reward_components": r.reward_components,
        })

    report = {
        "run_metadata": {
            "run_id": f"20260827_review2_episode_seed{args.seed}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "n_episodes": total,
            "defender": "FixtureModelAdapter (ReviewOneBaselineDefender wrapper)",
            "gate_version": "GATE-001 through GATE-009",
            "dataset_version": "review1-fixture-v2",
            "scope": "fixture episode harness; not a trained-LLM or production-security result",
        },
        "aggregate": {
            "total_episodes": total,
            "attack_episodes": len(attack_results),
            "benign_episodes": len(benign_results),
            "avg_total_reward": avg_reward,
            "gate_block_count": gate_blocks,
            "pre_gate_unsafe_count": pre_unsafe,
            "per_family_avg_reward": per_family_avg,
        },
        "capability_retention": {
            "total_probes": retention.total_probes if retention else 0,
            "passed": retention.passed if retention else 0,
            "retention_score": retention.retention_score if retention else 0.0,
            "regression_flagged": retention.regression_flagged if retention else True,
        },
        "episodes": episode_traces,
        "disclaimer": (
            "All results are from a deterministic rule-based fixture baseline. "
            "They validate the enforcement environment only. "
            "Do not present as trained-LLM or production-security results."
        ),
    }

    report_json = json.dumps(report, indent=2) + "\n"

    if args.dry_run:
        print("[dry-run] Episode run complete — no files written.")
        print(report_json)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "review2_episode_report.json"
    report_path.write_text(report_json, encoding="utf-8")

    print(f"Report written to {report_path}")
    print()
    print(f"  Episodes      : {total}")
    print(f"  Avg reward    : {avg_reward:.4f}")
    print(f"  Gate blocks   : {gate_blocks}")
    print(f"  Pre-gate unsafe: {pre_unsafe}")
    print(f"  Retention score: {retention.retention_score:.3f}" if retention else "  Retention: N/A")
    if retention and retention.regression_flagged:
        print("  *** CAPABILITY REGRESSION FLAGGED ***")


if __name__ == "__main__":
    main()
