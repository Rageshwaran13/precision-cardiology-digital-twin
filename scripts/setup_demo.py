"""Recreate the synthetic demonstration data and train the model.

Run from any working directory with:
    python scripts/setup_demo.py

The script executes each project stage in sequence and stops on the first error.
All generated datasets and model artifacts remain local and are git-ignored.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("Generate synthetic EHR data", "src/data/generate_ehr.py"),
    ("Generate synthetic wearable data", "src/data/generate_wearable.py"),
    ("Merge data and engineer features", "src/data/merge_datasets.py"),
    ("Train and evaluate the model", "src/model/train.py"),
]


def main() -> int:
    for label, relative_script in STAGES:
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(ROOT / relative_script)],
                cwd=ROOT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"Setup stopped: {relative_script} exited with code {exc.returncode}.",
                file=sys.stderr,
            )
            return exc.returncode or 1
    print("\nSetup complete. Start the dashboard with:")
    print("  streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
