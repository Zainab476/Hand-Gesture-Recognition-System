"""Model definition and loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import tensorflow as tf

from utils import LANDMARK_DIM


def build_model(num_classes: int, input_dim: int = LANDMARK_DIM) -> tf.keras.Model:
    """Build a dense neural network classifier."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_metadata(path: Path, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(model_path: Path, scaler_path: Path, encoder_path: Path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return model, scaler, label_encoder


def predict_proba(model: tf.keras.Model, scaler, landmarks: np.ndarray) -> np.ndarray:
    x = scaler.transform(landmarks.reshape(1, -1))
    return model.predict(x, verbose=0)[0]
