# Presentation Content: Real-Time Hand Gesture Recognition

## Slide 1 — Title
- ITML605: Robotics and Computer Vision
- Final Project: Real-Time Hand Gesture Recognition
- Student name / ID
- Instructor: Dr. Alauddin Al-Omary

## Slide 2 — Problem Statement
- Need robust, real-time contactless gesture interaction.
- Build a complete pipeline from data collection to live recognition.
- Support 8 static hand gestures for practical HCI/robotics control.

## Slide 3 — Objectives
- Detect hand from webcam in real time.
- Extract hand landmarks reliably.
- Classify gestures with deep learning.
- Show live label + confidence + FPS.
- Provide complete training/testing workflow.

## Slide 4 — Workflow
1. Capture video frame.
2. Detect hand landmarks (MediaPipe).
3. Normalize and scale features.
4. Predict gesture with trained model.
5. Display output on frame.

## Slide 5 — System Architecture
- Input: Webcam stream
- CV module: MediaPipe Hands
- Feature engineering: 63 normalized landmarks
- ML module: TensorFlow dense classifier
- Output: live gesture, confidence, FPS

## Slide 6 — Dataset
- 8 gesture classes:
  - Thumbs Up, Thumbs Down, Peace Sign, Stop Palm
  - OK Sign, Fist, Open Hand, Pointing
- Two collection methods:
  - Webcam capture script
  - Synthetic dataset generator for reproducibility
- CSV format: `f0..f62 + label`

## Slide 7 — Implementation
- `dataset_creator.py`: data collection/generation
- `train.py`: train/augment/evaluate/save model
- `main.py`: real-time detection + overlay
- `utils.py` and `model.py`: reusable utilities and architecture

## Slide 8 — Training and Validation
- Train/test split (stratified).
- Augmentation: noise + mirroring.
- Metrics: accuracy, precision, recall, F1-score.
- Confusion matrix generated and saved.

## Slide 9 — Results
- Test accuracy: **86.87%** (TensorFlow evaluation)
- Macro F1-score: **85.94%**
- Strong classes (F1 = 1.00): fist, ok_sign, peace_sign, pointing, thumbs_down, thumbs_up
- Main confusion: open_hand vs stop_palm (similar shape in normalized landmark space)
- Show confusion matrix image from `artifacts/confusion_matrix.png`

## Slide 10 — Demo Explanation
- Run `python main.py`.
- Show live detection and gesture prediction.
- Demonstrate multiple gestures in front of webcam.
- Explain confidence behavior and FPS performance.

## Slide 11 — Challenges and Improvements
- Lighting and occlusion sensitivity.
- Intra-user variation.
- Improve with larger real dataset and temporal models.

## Slide 12 — Conclusion
- Full end-to-end real-time gesture recognition delivered.
- Meets CV + deep learning + software engineering requirements.
- Ready for submission, Q&A, and live demo.
