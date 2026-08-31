"""Convert the Defender's 10-field SFT corpus into flat text records for train_sft.py.

Usage
-----
    python3 scripts/format_sft_for_training.py [--split train|validation|heldout] [--output-dir DIR] [--dry-run]

train_sft.py (via TRL's SFTTrainer) expects each JSONL line to be
``{"text": "..."}``. The Defender corpus in data/sft/*.jsonl instead stores
structured fields (user_task, untrusted_content, target, safety_rationale,
etc.) for schema validation, split-discipline checks, and audit purposes.
This script keeps the structured files as the source of truth and only
projects them into the flat prompt/response format the trainer needs.

It never modifies the original data/sft/*.jsonl files.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defender_policy.sft_data import load_jsonl  # noqa: E402


def to_text_record(row: dict) -> dict:
    """Project one structured SFT record into a single training-text turn.

    Untrusted content is included in the user turn (as the Defender would
    actually see it at inference time) so the model learns to treat it as
    data, not instructions, rather than training on a sanitized prompt it
    will never see in practice.
    """
    user_lines = [row["user_task"]]
    if row["untrusted_content"]:
        user_lines.append(f"[untrusted content]\n{row['untrusted_content']}")
    user_turn = "\n".join(user_lines)
    text = f"User: {user_turn}\nAssistant: {row['target']}"
    return {"text": text}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "heldout"],
        default="train",
        help="Which SFT split to convert (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "sft"),
        help="Directory for the formatted output file (default: data/sft/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert and print the first record without writing a file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = PROJECT_ROOT / "data" / "sft" / f"{args.split}.jsonl"
    rows = load_jsonl(source_path)
    records = [to_text_record(row) for row in rows]

    if args.dry_run:
        print(f"[dry-run] Converted {len(records)} records from {source_path}. First record:")
        print(json.dumps(records[0], indent=2) if records else "(empty)")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_formatted.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} formatted records to {output_path}")


if __name__ == "__main__":
    main()
