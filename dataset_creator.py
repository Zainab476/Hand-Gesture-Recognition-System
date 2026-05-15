"""Dataset creation utility for real and synthetic hand gesture landmarks."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from utils import GESTURE_LABELS, LANDMARK_DIM, normalize_landmarks, pretty_label


def synthetic_landmark_templates() -> Dict[str, np.ndarray]:
    """Create deterministic base hand templates for each gesture."""
    rng = np.random.default_rng(7)
    base = np.zeros((21, 3), dtype=np.float32)

    # Build a generic open hand scaffold.
    finger_bases = [1, 5, 9, 13, 17]
    for i, idx in enumerate(finger_bases):
        x_offset = (i - 2) * 0.12
        base[idx] = [x_offset, 0.05, 0.0]
        base[idx + 1] = [x_offset, -0.10, -0.02]
        base[idx + 2] = [x_offset, -0.22, -0.03]
        base[idx + 3] = [x_offset, -0.35, -0.04]
    base[0] = [0.0, 0.0, 0.0]

    templates: Dict[str, np.ndarray] = {}

    open_hand = base.copy()
    templates["open_hand"] = open_hand

    fist = base.copy()
    for idx in [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]:
        fist[idx, 1] = 0.02 + rng.normal(0, 0.01)
    templates["fist"] = fist

    thumbs_up = fist.copy()
    thumbs_up[2] = [-0.20, -0.06, 0.0]
    thumbs_up[3] = [-0.24, -0.15, 0.0]
    thumbs_up[4] = [-0.28, -0.28, 0.0]
    templates["thumbs_up"] = thumbs_up

    thumbs_down = thumbs_up.copy()
    thumbs_down[:, 1] *= -1
    templates["thumbs_down"] = thumbs_down

    peace = fist.copy()
    for idx in [6, 7, 8, 10, 11, 12]:
        peace[idx, 1] -= 0.25
    templates["peace_sign"] = peace

    stop_palm = open_hand.copy()
    stop_palm[:, 2] += 0.06
    templates["stop_palm"] = stop_palm

    ok_sign = open_hand.copy()
    ok_sign[4] = ok_sign[8] + np.array([0.0, 0.01, 0.0], dtype=np.float32)
    ok_sign[6:9, 1] += 0.08
    templates["ok_sign"] = ok_sign

    pointing = fist.copy()
    for idx in [6, 7, 8]:
        pointing[idx, 1] -= 0.28
    templates["pointing"] = pointing

    return templates


def augment_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = sample.reshape(21, 3).copy()
    arr += rng.normal(0, 0.012, size=arr.shape)

    scale = rng.uniform(0.9, 1.1)
    arr *= scale

    theta = rng.uniform(-0.25, 0.25)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    arr = arr @ rot.T

    return normalize_landmarks(arr.reshape(-1))


def generate_synthetic_dataset(output_csv: Path, samples_per_class: int = 400) -> None:
    templates = synthetic_landmark_templates()
    rng = np.random.default_rng(42)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [f"f{i}" for i in range(LANDMARK_DIM)] + ["label"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label in GESTURE_LABELS:
            template = templates[label].reshape(-1)
            for _ in range(samples_per_class):
                row = augment_sample(template, rng).tolist() + [label]
                writer.writerow(row)

    print(f"Synthetic dataset created at {output_csv} with {samples_per_class * len(GESTURE_LABELS)} samples.")


def collect_webcam_dataset(
    output_csv: Path,
    samples_per_gesture: int,
    camera_index: int = 0,
) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("MediaPipe is required for webcam collection. Install requirements first.") from exc

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: List[List[str]] = []
    header = [f"f{i}" for i in range(LANDMARK_DIM)] + ["label"]
    if output_csv.exists():
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows and rows[0] == header:
                existing_rows = rows[1:]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or camera index.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    counts = {g: 0 for g in GESTURE_LABELS}
    current_label_idx = 0

    print("Instructions:")
    print("  - Press SPACE to capture current hand for selected gesture")
    print("  - Press N for next gesture")
    print("  - Press Q to save and quit")

    rows = existing_rows
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            landmarks_flat = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                pts = []
                for lm in hand.landmark:
                    pts.extend([lm.x, lm.y, lm.z])
                landmarks_flat = normalize_landmarks(pts)

            current_label = GESTURE_LABELS[current_label_idx]
            cv2.putText(
                frame,
                f"Collecting: {pretty_label(current_label)} ({counts[current_label]}/{samples_per_gesture})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "SPACE: capture  N: next gesture  Q: save & quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Dataset Creator", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if landmarks_flat is None:
                    print("No hand detected. Try again.")
                elif counts[current_label] >= samples_per_gesture:
                    print(f"{current_label} is already complete.")
                else:
                    rows.append([*map(str, landmarks_flat.tolist()), current_label])
                    counts[current_label] += 1
                    print(f"Captured {current_label}: {counts[current_label]}/{samples_per_gesture}")

            elif key == ord("n"):
                current_label_idx = (current_label_idx + 1) % len(GESTURE_LABELS)

            elif key == ord("q"):
                break

            if all(v >= samples_per_gesture for v in counts.values()):
                print("All gestures collected. Saving dataset.")
                break

    cap.release()
    cv2.destroyAllWindows()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Dataset saved to {output_csv} with {len(rows)} samples.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hand gesture landmark dataset")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "webcam"],
        default="synthetic",
        help="Dataset creation mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gesture_landmarks.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=400,
        help="Samples per class for synthetic generation",
    )
    parser.add_argument(
        "--samples-per-gesture",
        type=int,
        default=200,
        help="Samples per gesture for webcam collection",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "synthetic":
        generate_synthetic_dataset(args.output, args.samples_per_class)
    else:
        collect_webcam_dataset(args.output, args.samples_per_gesture, args.camera_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
