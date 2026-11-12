"""Train and evaluate the Hybrid GCN-Transformer Encoder landslide model.

Example (run from the repository root):
    python src/train.py

Default hyperparameters are those used in the notebook and the manuscript.
"""

import argparse
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (evaluate, load_dataset, preprocess, print_metrics,  # noqa: E402
                        save_outputs)
from models import build_gcn_transformer_model  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "GCN-Tranformer-Hybrid Model"
DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "soil_data2.csv")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA, help="CSV file (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5, help="Adam learning rate")
    p.add_argument("--seed", type=int,
                   help="global random seed; the notebook was run without one")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs"),
                   help="output directory (default: %(default)s)")
    p.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    return p.parse_args(argv)


def run(args):
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {args.data}")
    df = load_dataset(args.data)
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    model = build_gcn_transformer_model(input_shape=X_train.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=args.epochs, batch_size=args.batch_size,
                        verbose=args.verbose)

    y_prob = model.predict(X_test, verbose=0)
    metrics = evaluate(y_test, y_prob)
    print_metrics(metrics)

    save_outputs(args.out, history, y_test, y_prob, metrics, MODEL_NAME)
    model.save(os.path.join(args.out, "model.keras"))
    print(f"\nOutputs saved to {args.out}")
    return metrics


if __name__ == "__main__":
    run(parse_args())
