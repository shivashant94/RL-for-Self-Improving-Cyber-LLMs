"""Capability-retention analysis script for Review 3.

Runs the capability-retention probe bank against the fixture baseline and
produces structured per-category and per-probe breakdown reports (JSON + CSV).

Usage
-----
    python3 scripts/run_capability_retention_analysis.py
            [--checkpoint ID] [--threshold FLOAT]
            [--output-dir DIR] [--dry-run]

Outputs
-------
    <output-dir>/capability_retention_<checkpoint>.json   full JSON report
    <output-dir>/capability_retention_<checkpoint>.csv    per-probe CSV

Design rules
------------
- Uses built-in FIXTURE_PROBES only. No held-out data tuned on.
- Gate is active (same as during training).
- Results are tagged with scope disclaimer.
- CSV is for presentation/plotting; JSON is the authoritative record.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defender_policy.capability_retention import (  # noqa: E402
    FIXTURE_PROBES,
    CapabilityRetentionReport,
    run_capability_probe,
)
from defender_policy.model_adapter import FixtureModelAdapter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capability-retention analysis for a Defender checkpoint."
    )
    parser.add_argument(
        "--checkpoint", default="fixture_baseline",
        help="Checkpoint identifier (default: fixture_baseline).",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="Minimum acceptable retention score (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "reports"),
        help="Directory for output files (default: reports/).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run analysis but do not write files.")
    return parser.parse_args()


def _build_report(report: CapabilityRetentionReport, checkpoint_id: str) -> dict:
    """Serialise a CapabilityRetentionReport to a JSON-ready dict."""
    return {
        "checkpoint_id": checkpoint_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_probes": report.total_probes,
        "passed": report.passed,
        "failed": report.failed,
        "retention_score": report.retention_score,
        "threshold": report.threshold,
        "regression_flagged": report.regression_flagged,
        "per_category_pass_rate": report.per_category_pass_rate,
        "per_probe": [
            {
                "probe_id": r.probe_id,
                "probe_category": r.probe_category,
                "expected_status": r.expected_status,
                "actual_status": r.actual_status,
                "passed": r.passed,
                "note": r.note,
            }
            for r in report.results
        ],
        "scope": report.scope,
        "disclaimer": (
            "Fixture-based capability probing only. "
            "Scores reflect rule-based baseline behaviour; "
            "replace FixtureModelAdapter with SFTModelAdapter "
            "once base model is confirmed."
        ),
    }


def _write_csv(report: CapabilityRetentionReport, path: Path) -> None:
    """Write per-probe CSV (useful for presentation plots)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["probe_id", "probe_category", "expected_status",
                        "actual_status", "passed"],
        )
        writer.writeheader()
        for r in report.results:
            writer.writerow({
                "probe_id":        r.probe_id,
                "probe_category":  r.probe_category,
                "expected_status": r.expected_status,
                "actual_status":   r.actual_status,
                "passed":          r.passed,
            })


def _print_summary(report: CapabilityRetentionReport, checkpoint_id: str) -> None:
    print(f"\n=== Capability Retention — {checkpoint_id} ===")
    print(f"  Score            : {report.retention_score:.3f}  "
          f"({report.passed}/{report.total_probes} probes passed)")
    print(f"  Threshold        : {report.threshold:.3f}")
    if report.regression_flagged:
        print("  *** CAPABILITY REGRESSION FLAGGED ***")
    else:
        print("  Regression       : No")
    print()
    print("  Per-category pass rate:")
    for cat, rate in sorted(report.per_category_pass_rate.items()):
        bar = "✓" if rate == 1.0 else ("△" if rate > 0.0 else "✗")
        print(f"    {bar} {cat:<30s} {rate:.2f}")
    print()
    print("  Per-probe results:")
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    [{status}] {r.probe_id:<10} {r.probe_category:<30} "
              f"expected={r.expected_status:<10} got={r.actual_status}")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    # Use FixtureModelAdapter until a real checkpoint is available.
    # Replace with SFTModelAdapter(checkpoint_path=...) once confirmed.
    defender = None  # run_capability_probe defaults to ReviewOneBaselineDefender

    report = run_capability_probe(defender=defender, threshold=args.threshold)
    _print_summary(report, args.checkpoint)

    report_dict = _build_report(report, args.checkpoint)

    if args.dry_run:
        print("[dry-run] Analysis complete — no files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = args.checkpoint.replace("/", "_").replace(" ", "_")

    json_path = output_dir / f"capability_retention_{safe_id}.json"
    csv_path  = output_dir / f"capability_retention_{safe_id}.csv"

    json_path.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")
    _write_csv(report, csv_path)

    print(f"\nJSON report : {json_path}")
    print(f"CSV  report : {csv_path}")


if __name__ == "__main__":
    main()
