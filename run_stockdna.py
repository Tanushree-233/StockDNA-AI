import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


STEPS = [
    ("Download news", "scripts.collectors.download_news"),
    ("Download earnings events", "scripts.collectors.download_earnings_events"),
    (
        "Build internal dataset",
        "scripts.processors.build_internal_dataset",
    ),
    (
        "Build earnings-event dataset",
        "scripts.processors.build_earnings_event_dataset",
    ),
    (
        "Build final ML dataset",
        "scripts.processors.build_final_ml_dataset",
    ),
    (
        "Run live prediction",
        "scripts.ml.predict.live_prediction",
    ),
]


def run_step(name, module):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
    )

    if result.returncode != 0:
        print(f"\nFAILED: {name}")
        sys.exit(result.returncode)

    print(f"\nPASSED: {name}")


def main():

    print("=" * 70)
    print("STOCKDNA-AI PRODUCTION PIPELINE")
    print("=" * 70)

    for name, module in STEPS:
        run_step(name, module)

    print("\n" + "=" * 70)
    print("STOCKDNA-AI PIPELINE COMPLETE")
    print("=" * 70)

    print("\nLatest predictions:")

    print(
        ROOT
        / "data"
        / "processed"
        / "internal"
        / "live_predictions.csv"
    )


if __name__ == "__main__":
    main()