"""Training script for hand gesture classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

from model import build_model, save_training_metadata
from utils import GESTURE_LABELS, LANDMARK_DIM


def load_dataset(dataset_csv: Path):
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    expected_cols = {f"f{i}" for i in range(LANDMARK_DIM)} | {"label"}
    if not expected_cols.issubset(df.columns):
        raise ValueError("Dataset CSV does not have expected landmark feature columns.")

    x = df[[f"f{i}" for i in range(LANDMARK_DIM)]].values.astype(np.float32)
    y = df["label"].astype(str).values
    return x, y


def augment_training_data(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)

    noise = rng.normal(0, 0.01, size=x_train.shape).astype(np.float32)
    x_noise = x_train + noise

    x_mirror = x_train.copy()
    x_mirror = x_mirror.reshape(-1, 21, 3)
    x_mirror[:, :, 0] *= -1
    x_mirror = x_mirror.reshape(-1, LANDMARK_DIM)

    x_aug = np.vstack([x_train, x_noise, x_mirror])
    y_aug = np.concatenate([y_train, y_train, y_train])
    return x_aug, y_aug


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hand gesture recognition model")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/gesture_landmarks.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    x, y = load_dataset(args.dataset)
    missing = sorted(set(GESTURE_LABELS) - set(np.unique(y)))
    if missing:
        raise ValueError(f"Dataset is missing required gestures: {missing}")

    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    x_train_aug, y_train_aug = augment_training_data(x_train, y_train_raw)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_aug)
    x_test_scaled = scaler.transform(x_test)

    label_encoder = LabelEncoder()
    y_train_idx = label_encoder.fit_transform(y_train_aug)
    y_test_idx = label_encoder.transform(y_test_raw)

    y_train = tf.keras.utils.to_categorical(y_train_idx)
    y_test = tf.keras.utils.to_categorical(y_test_idx)

    model = build_model(num_classes=len(label_encoder.classes_), input_dim=x_train_scaled.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    history = model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test_scaled, y_test, verbose=0)
    probs = model.predict(x_test_scaled, verbose=0)
    y_pred_idx = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test_idx, y_pred_idx)
    report = classification_report(
        y_test_idx,
        y_pred_idx,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_idx, y_pred_idx)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "gesture_model.keras"
    scaler_path = args.artifacts_dir / "scaler.pkl"
    encoder_path = args.artifacts_dir / "label_encoder.pkl"
    metrics_path = args.artifacts_dir / "metrics.json"
    history_path = args.artifacts_dir / "history.json"
    cm_path = args.artifacts_dir / "confusion_matrix.png"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "sklearn_accuracy": float(acc),
        "classes": label_encoder.classes_.tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    plot_confusion_matrix(cm, label_encoder.classes_.tolist(), cm_path)

    save_training_metadata(
        args.artifacts_dir / "training_metadata.json",
        {
            "dataset": str(args.dataset),
            "epochs_requested": args.epochs,
            "epochs_ran": len(history.history.get("loss", [])),
            "batch_size": args.batch_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
    )

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Metrics: {metrics_path}")
    print(f"Confusion matrix image: {cm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
