"""Real-time hand gesture recognition app."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from model import load_artifacts, predict_proba
from utils import FPSCounter, draw_prediction_overlay, extract_landmarks_from_results, pretty_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time hand gesture recognition")
    parser.add_argument("--model", type=Path, default=Path("artifacts/gesture_model.keras"))
    parser.add_argument("--scaler", type=Path, default=Path("artifacts/scaler.pkl"))
    parser.add_argument("--encoder", type=Path, default=Path("artifacts/label_encoder.pkl"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in [args.model, args.scaler, args.encoder]:
        if not path.exists():
            raise SystemExit(
                f"Required artifact not found: {path}. Run train.py first to generate model files."
            )

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("MediaPipe is not installed. Run: pip install -r requirements.txt") from exc

    model, scaler, label_encoder = load_artifacts(args.model, args.scaler, args.encoder)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or camera index.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    fps_counter = FPSCounter()
    last_label = "No Hand"
    last_conf = 0.0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            landmarks = extract_landmarks_from_results(results)
            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            if landmarks is not None:
                probs = predict_proba(model, scaler, landmarks)
                best_idx = int(np.argmax(probs))
                last_label = str(label_encoder.inverse_transform([best_idx])[0])
                last_conf = float(probs[best_idx])
            else:
                last_label = "No Hand"
                last_conf = 0.0

            fps = fps_counter.update()
            draw_prediction_overlay(frame, pretty_label(last_label), last_conf, fps)
            cv2.putText(
                frame,
                "Press Q to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
