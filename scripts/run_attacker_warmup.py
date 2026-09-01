import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent

    training_script = project_root / "src" / "train_sft.py"

    dataset_path = (
        project_root
        / "data"
        / "attacker"
        / "formatted_attacks.jsonl"
    )

    output_dir = (
        project_root
        / "attacker_checkpoints"
    )

    if not training_script.exists():
        raise FileNotFoundError(
            f"Training script not found: {training_script}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Formatted attacker dataset not found: {dataset_path}"
        )

    print("=" * 60)
    print("STARTING ATTACKER WARMUP")
    print("=" * 60)

    print(f"\nDataset: {dataset_path}")
    print(f"Output directory: {output_dir}\n")

    command = [
        sys.executable,
        str(training_script),
        "--dataset_path",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
    ]

    subprocess.run(command, check=True)

    print("\n" + "=" * 60)
    print("ATTACKER WARMUP COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()