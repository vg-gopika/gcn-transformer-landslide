"""Quick test: trains the model for a few epochs on the small example file
``data/sample/sample_data.csv`` and checks that the full pipeline runs end
to end.

Run from the repository root:
    python tests/quick_test.py

Takes well under a minute on a laptop CPU. It checks that the code works;
it does NOT reproduce the accuracy reported in the paper (use src/train.py
on the full dataset for that).
"""

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from train import parse_args, run  # noqa: E402

SAMPLE_CSV = os.path.join(REPO_ROOT, "data", "sample", "sample_data.csv")
EXPECTED_FILES = ["metrics.txt", "test_predictions.csv", "accuracy.png", "loss.png",
                  "confusion_matrix.png", "roc_curve.png", "model.keras"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as out_dir:
        args = parse_args(["--data", SAMPLE_CSV, "--epochs", "3", "--seed", "42",
                           "--out", out_dir, "--verbose", "0"])
        metrics = run(args)

        n_test = metrics["confusion_matrix"].sum()
        assert n_test == 60, f"expected 60 test samples, got {n_test}"
        assert 0 <= metrics["accuracy"] <= 100
        assert np.isfinite(metrics["auc"])
        for name in EXPECTED_FILES:
            assert os.path.exists(os.path.join(out_dir, name)), f"missing output {name}"

    print(f"\n[PASS] pipeline ran, accuracy on sample = {metrics['accuracy']:.2f} %")
    print("Quick test passed.")
