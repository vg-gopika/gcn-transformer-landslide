"""Data loading, preprocessing and evaluation helpers.

The steps mirror the notebook in ``notebooks/``:
  1. drop the label and the spatial coordinates (Longitude, Latitude)
  2. z-score standardisation of every feature
  3. expand each sample into a fixed-length sequence of T = 10 steps
  4. stratified 80:20 train/test split (random_state = 42)
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_COL = "Label"
COORD_COLS = ["Longitude", "Latitude"]
CLASS_NAMES = ["No Risk", "Land Slide"]
SEQ_LEN = 10


def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. The full dataset is not included in this "
            "repository (see 'Data availability' in README.md). Place it at "
            "data/soil_data2.csv or pass --data PATH.")
    df = pd.read_csv(csv_path)
    missing = [c for c in [LABEL_COL] + COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df


def preprocess(df, seq_len=SEQ_LEN, test_size=0.2, random_state=42):
    """Return X_train, X_test, y_train, y_test as used in the notebooks."""
    X = df.drop(columns=[LABEL_COL] + COORD_COLS).values
    y = df[LABEL_COL].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Repeat each feature vector over seq_len time steps -> (N, T, F)
    X_seq = np.repeat(X_scaled[:, np.newaxis, :], seq_len, axis=1)

    return train_test_split(X_seq, y, test_size=test_size, random_state=random_state,
                            stratify=y)


def evaluate(y_true, y_prob, threshold=0.5):
    """Compute the metrics reported in the paper (weighted averages, in %)."""
    y_true = np.asarray(y_true).astype(int).flatten()
    y_prob = np.asarray(y_prob).flatten()
    y_pred = (y_prob > threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred) * 100
    return {
        "accuracy": acc,
        "precision": precision_score(y_true, y_pred, average="weighted") * 100,
        "recall": recall_score(y_true, y_pred, average="weighted") * 100,
        "f1": f1_score(y_true, y_pred, average="weighted") * 100,
        "misclassification": 100 - acc,
        "auc": auc(fpr, tpr),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "report": classification_report(y_true, y_pred, labels=[0, 1],
                                        target_names=CLASS_NAMES, zero_division=0),
        "y_pred": y_pred,
        "roc": (fpr, tpr),
    }


def print_metrics(metrics):
    print(metrics["report"])
    print(f"Accuracy Score          : {metrics['accuracy']:.4f} %")
    print(f"Precision Score         : {metrics['precision']:.4f} %")
    print(f"Recall Score            : {metrics['recall']:.4f} %")
    print(f"F1 Score                : {metrics['f1']:.4f} %")
    print(f"Misclassification Score : {metrics['misclassification']:.4f} %")
    print(f"ROC AUC                 : {metrics['auc']:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted) [No Risk, Land Slide]:")
    print(metrics["confusion_matrix"])


def save_outputs(out_dir, history, y_true, y_prob, metrics, model_label):
    """Save metrics, predictions and the main figures to ``out_dir``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(metrics["report"] + "\n")
        for k in ["accuracy", "precision", "recall", "f1", "misclassification"]:
            f.write(f"{k}: {metrics[k]:.4f} %\n")
        f.write(f"auc: {metrics['auc']:.4f}\n")
        f.write(f"confusion_matrix:\n{metrics['confusion_matrix']}\n")

    label_map = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}
    pd.DataFrame({
        "Original Label": pd.Series(np.asarray(y_true).astype(int)).map(label_map),
        "Predicted Label": pd.Series(metrics["y_pred"]).map(label_map),
        "Predicted Probability": np.asarray(y_prob).flatten(),
    }).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    h = history.history
    for key, title in [("accuracy", "Model Accuracy"), ("loss", "Model Loss")]:
        plt.figure()
        plt.plot(h[key], label=f"Train {key.title()}")
        plt.plot(h[f"val_{key}"], label=f"Val {key.title()}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(key.title())
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    cm = pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix of Land Slide Prediction")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fpr, tpr = metrics["roc"]
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"{model_label} (AUC = {metrics['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chances")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
