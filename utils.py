"""Utility functions for hand gesture recognition."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

GESTURE_LABELS: List[str] = [
    "thumbs_up",
    "thumbs_down",
    "peace_sign",
    "stop_palm",
    "ok_sign",
    "fist",
    "open_hand",
    "pointing",
]

LANDMARK_DIM = 21 * 3


@dataclass
class PredictionResult:
    label: str
    confidence: float
    fps: float


class FPSCounter:
    """Simple FPS calculator."""

    def __init__(self) -> None:
        self._last_time = time.time()
        self._fps = 0.0

    def update(self) -> float:
        now = time.time()
        diff = now - self._last_time
        self._last_time = now
        if diff > 0:
            self._fps = 1.0 / diff
        return self._fps


def normalize_landmarks(landmarks_flat: Sequence[float]) -> np.ndarray:
    """Normalize 21x3 landmarks to translation and scale invariance."""
    arr = np.asarray(landmarks_flat, dtype=np.float32)
    if arr.size != LANDMARK_DIM:
        raise ValueError(f"Expected {LANDMARK_DIM} values, got {arr.size}")

    arr = arr.reshape(21, 3)
    wrist = arr[0].copy()
    arr = arr - wrist

    max_dist = np.max(np.linalg.norm(arr, axis=1))
    if max_dist > 1e-8:
        arr = arr / max_dist

    return arr.reshape(-1)


def extract_landmarks_from_results(results) -> Optional[np.ndarray]:
    """Extract normalized landmarks from MediaPipe Hands output."""
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    pts = []
    for lm in hand_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return normalize_landmarks(pts)


def draw_prediction_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    fps: float,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw prediction details on the frame."""
    text = f"Gesture: {label} ({confidence:.2%})"
    fps_text = f"FPS: {fps:.1f}"
    cv2.rectangle(frame, (10, 10), (520, 90), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, fps_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def pretty_label(label: str) -> str:
    return label.replace("_", " ").title()
