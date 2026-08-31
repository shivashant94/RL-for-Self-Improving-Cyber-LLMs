"""CLI script: evaluate a Defender checkpoint and write structured reports.

Usage
-----
    python3 scripts/evaluate_checkpoint.py [--checkpoint ID] [--allow-heldout]
                                            [--seed SEED] [--output-dir DIR]
                                            [--dry-run]

Outputs
-------
    <output-dir>/checkpoint_eval_<checkpoint_id>.json   structured metrics
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defender_policy.evaluate_checkpoint import evaluate_checkpoint  # noqa: E402
from defender_policy.model_adapter import FixtureModelAdapter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Defender checkpoint on in-distribution and (optionally) held-out cases."
    )
    parser.add_argument(
        "--checkpoint", default="fixture_baseline",
        help="Checkpoint identifier (default: fixture_baseline).",
    )
    parser.add_argument(
        "--allow-heldout", action="store_true",
        help=(
            "Enable held-out OOD evaluation. "
            "Use ONLY at final Review 3 — not during hyperparameter search."
        ),
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"),
                        help="Directory for output report (default: reports/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate but do not write any files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    # Always use FixtureModelAdapter until a real checkpoint is available.
    # Replace with SFTModelAdapter(checkpoint_path=...) once base model is confirmed.
    adapter = FixtureModelAdapter()

    result = evaluate_checkpoint(
        checkpoint_id=args.checkpoint,
        adapter=adapter,
        allow_heldout=args.allow_heldout,
        seed=args.seed,
    )

    # ── Format output ────────────────────────────────────────────────────
    def _slice_dict(s) -> dict:
        if s is None:
            return None
        return {
            "slice_name": s.slice_name,
            "n_cases": s.n_cases,
            "attack_success_rate": s.attack_success_rate,
            "benign_task_success_rate": s.benign_task_success_rate,
            "false_refusal_rate": s.false_refusal_rate,
            "tool_block_rate": s.tool_block_rate,
            "per_family_asr": s.per_family_asr,
            "avg_total_reward": s.avg_total_reward,
            "evaluation_scope": s.evaluation_scope,
        }

    output = {
        "checkpoint_id": result.checkpoint_id,
        "seed": result.seed,
        "dataset_version": result.dataset_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allow_heldout": result.allow_heldout,
        "evaluation_scope": result.evaluation_scope,
        "in_distribution": _slice_dict(result.in_distribution),
        "heldout_ood": _slice_dict(result.heldout_ood),
        "disclaimer": (
            "Fixture results only. Replace FixtureModelAdapter with "
            "SFTModelAdapter(checkpoint_path=...) once base model is confirmed. "
            "Do not report fixture results as trained-model performance."
        ),
    }

    report_json = json.dumps(output, indent=2) + "\n"

    if args.dry_run:
        print("[dry-run] Evaluation complete — no files written.")
        _print_summary(result)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = args.checkpoint.replace("/", "_").replace(" ", "_")
    report_path = output_dir / f"checkpoint_eval_{safe_id}.json"
    report_path.write_text(report_json, encoding="utf-8")
    print(f"Report written to {report_path}")
    _print_summary(result)


def _print_summary(result) -> None:
    i = result.in_distribution
    print()
    print(f"=== Checkpoint: {result.checkpoint_id} ===")
    print(f"  In-distribution ({i.n_cases} cases):")
    print(f"    ASR              : {i.attack_success_rate:.4f}")
    print(f"    Benign success   : {i.benign_task_success_rate:.4f}")
    print(f"    False refusal    : {i.false_refusal_rate:.4f}")
    print(f"    Tool block rate  : {i.tool_block_rate:.4f}")
    print(f"    Avg total reward : {i.avg_total_reward:.4f}")
    print(f"    Per-family ASR   : {i.per_family_asr}")
    if result.heldout_ood:
        h = result.heldout_ood
        print(f"  Held-out OOD ({h.n_cases} cases):")
        print(f"    ASR              : {h.attack_success_rate:.4f}")
        print(f"    Benign success   : {h.benign_task_success_rate:.4f}")
        print(f"    False refusal    : {h.false_refusal_rate:.4f}")
        print(f"    Avg total reward : {h.avg_total_reward:.4f}")
    else:
        print("  Held-out OOD: not evaluated (pass --allow-heldout for Review 3 only)")


if __name__ == "__main__":
    main()
