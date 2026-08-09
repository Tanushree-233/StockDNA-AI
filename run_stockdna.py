import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Download earnings events",
     "scripts/collectors/download_earnings_events.py"),

    ("Build internal dataset",
     "scripts/processors/build_internal_dataset.py"),

    ("Build earnings-event dataset",
     "scripts/processors/build_earnings_event_dataset.py"),

    ("Build final ML dataset",
     "scripts/processors/build_final_ml_dataset.py"),

    ("Prepare live prediction",
     "scripts/ml/predict/run_live_prediction.py"),
]


def run_step(name, script):
    path = ROOT / script

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: Script not found:")
        print(path)
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT
    )

    if result.returncode != 0:
        print(f"\nFAILED: {name}")
        sys.exit(result.returncode)

    print(f"\nPASSED: {name}")


def main():
    print("=" * 70)
    print("STOCKDNA-AI PRODUCTION PIPELINE")
    print("=" * 70)

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 70)
    print("STOCKDNA-AI PIPELINE COMPLETE")
    print("=" * 70)

    print("\nLatest predictions:")
    print(
        ROOT /
        "data/processed/internal/live_predictions.csv"
    )


if __name__ == "__main__":
    main()