from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import LANDMARK_DIM, normalize_landmarks



def test_normalize_landmarks_shape_and_scale() -> None:
    sample = np.linspace(0.0, 1.0, LANDMARK_DIM)
    normalized = normalize_landmarks(sample)
    assert normalized.shape == (LANDMARK_DIM,)
    arr = normalized.reshape(21, 3)
    assert np.allclose(arr[0], np.zeros(3), atol=1e-6)
    max_norm = np.max(np.linalg.norm(arr, axis=1))
    assert 0.99 <= max_norm <= 1.01


def test_training_pipeline_generates_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "synthetic.csv"
    artifacts_dir = tmp_path / "artifacts"

    create_cmd = [
        sys.executable,
        str(REPO / "dataset_creator.py"),
        "--mode",
        "synthetic",
        "--output",
        str(dataset_path),
        "--samples-per-class",
        "60",
    ]
    subprocess.run(create_cmd, cwd=REPO, check=True, capture_output=True, text=True)

    train_cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "--dataset",
        str(dataset_path),
        "--artifacts-dir",
        str(artifacts_dir),
        "--epochs",
        "6",
        "--batch-size",
        "32",
    ]
    subprocess.run(train_cmd, cwd=REPO, check=True, capture_output=True, text=True)

    expected_files = [
        "gesture_model.keras",
        "scaler.pkl",
        "label_encoder.pkl",
        "metrics.json",
        "history.json",
        "confusion_matrix.png",
        "training_metadata.json",
    ]
    for filename in expected_files:
        assert (artifacts_dir / filename).exists(), f"Missing artifact: {filename}"

    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_accuracy"] >= 0.60
    assert len(metrics["classes"]) == 8
