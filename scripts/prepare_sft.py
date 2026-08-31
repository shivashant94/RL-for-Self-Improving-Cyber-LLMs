"""Prepare and validate the Defender SFT warm-up corpus.

Usage
-----
    python3 scripts/prepare_sft.py [--config PATH] [--output-dir DIR] [--dry-run]

Outputs
-------
    <output-dir>/sft_warmup_manifest.json   full validation manifest

The script validates the three-way split (train / validation / held-out) and
writes a reproducible manifest.  It never modifies the data files.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defender_policy.sft_data import prepare_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and manifest the SFT corpus.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "sft_warmup.json"),
        help="Path to SFT config JSON (default: configs/sft_warmup.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory for output manifest (default: reports/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    # Load config for metadata
    config: dict = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as fh:
            config = json.load(fh)

    train_path = PROJECT_ROOT / "data" / "sft" / "train.jsonl"
    validation_path = PROJECT_ROOT / "data" / "sft" / "validation.jsonl"
    heldout_path = PROJECT_ROOT / "data" / "sft" / "heldout.jsonl"

    manifest = prepare_manifest(
        train_path,
        validation_path,
        heldout_path=heldout_path if heldout_path.exists() else None,
    )

    # Annotate with run metadata
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["config_path"] = str(config_path)
    manifest["dataset_version"] = "sft-v2"
    manifest["seed"] = config.get("seed", "N/A")
    manifest["base_model"] = config.get("base_model", "PENDING")
    manifest["note"] = (
        "Fixture corpus only. Actual SFT training requires team-approved "
        "base model, training stack, and compute. Mark BLOCKED_EXTERNAL until available."
    )

    report_json = json.dumps(manifest, indent=2) + "\n"

    if args.dry_run:
        print("[dry-run] Manifest validated — no files written.")
        print(report_json)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sft_warmup_manifest.json"
    report_path.write_text(report_json, encoding="utf-8")
    print(f"Manifest written to {report_path}")
    print(report_json)


if __name__ == "__main__":
    main()
