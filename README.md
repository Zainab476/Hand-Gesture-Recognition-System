# Hand Gesture Recognition (ITML605 Final Project)

A complete university-grade **real-time Hand Gesture Recognition System** for Robotics and Computer Vision using **Python, OpenCV, MediaPipe, TensorFlow, NumPy, Scikit-learn, and Matplotlib**.

## 1) Project Features

- Real-time webcam hand detection.
- Hand landmark extraction (21 keypoints via MediaPipe).
- Gesture classification using deep learning.
- Multi-gesture support (8 gestures):
  - Thumbs Up
  - Thumbs Down
  - Peace Sign
  - Stop Palm
  - OK Sign
  - Fist
  - Open Hand
  - Pointing
- Live prediction and confidence overlay.
- Real-time FPS display.
- Dataset creation (webcam and synthetic).
- Preprocessing, augmentation, train/test split.
- Model save/load support.
- Evaluation with confusion matrix and classification report.
- Stable execution with explicit error handling.

## 2) Folder Structure

```text
Hand-gesture-recognition/
├── artifacts/                     # Saved model and evaluation artifacts
├── data/
│   ├── raw/                       # Optional raw captures
│   └── processed/                 # Landmark CSV datasets
├── docs/
│   ├── report.md                  # Full project report
│   └── presentation.md            # Slide-by-slide presentation content
├── outputs/                       # Runtime outputs (screenshots/videos if saved)
├── tests/
│   └── test_pipeline.py           # Automated tests
├── dataset_creator.py             # Dataset collection/generation
├── main.py                        # Real-time inference app
├── model.py                       # Model architecture and artifact loaders
├── train.py                       # Model training + evaluation
├── utils.py                       # Shared utilities
├── requirements.txt               # Python dependencies
└── README.md
```

## 3) Installation and Environment Setup

### Quick start on this workspace

If you are using the validated Windows workspace Python, install requirements with:

```powershell
D:/000000_coding-night_practices/coding-night_practice_23-11-2024/python_ml/python.exe -m pip install -r requirements.txt
```

Then run tests and the demo with:

```powershell
D:/000000_coding-night_practices/coding-night_practice_23-11-2024/python_ml/python.exe -m pytest -q
D:/000000_coding-night_practices/coding-night_practice_23-11-2024/python_ml/python.exe main.py --model artifacts_test/gesture_model.keras --scaler artifacts_test/scaler.pkl --encoder artifacts_test/label_encoder.pkl
```

### Option A: VS Code / Terminal (recommended)

```bash
# 1) Enter repository
cd /home/runner/work/Hand-gesture-recognition/Hand-gesture-recognition

# 2) Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3) Upgrade pip
python -m pip install --upgrade pip

# 4) Install requirements
pip install -r requirements.txt
```

In VS Code, you can also use the bundled launch profile **Run Hand Gesture Recognition** or the matching task in [.vscode/tasks.json](.vscode/tasks.json).

### Option B: PyCharm

1. Open project folder in PyCharm.
2. Create/select a Python interpreter (virtualenv).
3. Install dependencies from `requirements.txt`.

## 4) Requirements

- Python 3.10+ recommended.
- Webcam for real-time demo mode.
- Packages listed in `requirements.txt`.
- `mediapipe==0.10.14` is pinned for `mp.solutions` compatibility with `main.py`.

## 5) Dataset Handling

### Dataset source

This project supports two real dataset pipelines:

1. **Webcam data collection** (real user hand samples) using MediaPipe landmarks.
2. **Synthetic landmark generation** for fast reproducible training and testing.

### A) Create synthetic dataset (quick start)

```bash
python dataset_creator.py --mode synthetic --output data/processed/gesture_landmarks.csv --samples-per-class 400
```

### B) Create webcam dataset (for live demo quality)

```bash
python dataset_creator.py --mode webcam --output data/processed/gesture_landmarks.csv --samples-per-gesture 200 --camera-index 0
```

Controls in webcam collector:
- `SPACE`: capture sample for current gesture
- `N`: switch gesture
- `Q`: save and quit

### Preprocessing steps

- Convert 21 hand landmarks `(x, y, z)` to a 63-length vector.
- Normalize by:
  - Translation removal using wrist landmark as origin.
  - Scale normalization by maximum joint distance.
- Standardization using `StandardScaler` during training.

### Data augmentation

Applied on training split:
- Gaussian noise augmentation.
- Horizontal mirroring on x-axis.

### Train/test split

- Stratified split using `train_test_split`.
- Default: 80% train / 20% test.

## 6) Model Training

For a fast smoke test, the repository includes a synthetic dataset flow verified by the automated tests.

```bash
python train.py \
  --dataset data/processed/gesture_landmarks.csv \
  --artifacts-dir artifacts \
  --epochs 40 \
  --batch-size 32
```

Training outputs:
- `artifacts/gesture_model.keras`
- `artifacts/scaler.pkl`
- `artifacts/label_encoder.pkl`
- `artifacts/metrics.json`
- `artifacts/history.json`
- `artifacts/confusion_matrix.png`
- `artifacts/training_metadata.json`

## 7) Run Real-Time Gesture Recognition

```bash
python main.py --camera-index 0
```

If the model artifacts were created in `artifacts_test`, use the validated command below:

```powershell
D:/000000_coding-night_practices/coding-night_practice_23-11-2024/python_ml/python.exe main.py --model artifacts_test/gesture_model.keras --scaler artifacts_test/scaler.pkl --encoder artifacts_test/label_encoder.pkl
```

Expected display:
- Webcam stream with hand landmarks.
- Predicted gesture label.
- Confidence score.
- Real-time FPS.
- `Q` to quit.

## 8) Testing Commands

Run automated tests:

```bash
pytest -q
```

Optional syntax check:

```bash
python -m compileall .
```

## 9) File-by-File Explanation

- `dataset_creator.py`:
  - Creates datasets via webcam or synthetic generator.
  - Handles sampling controls and CSV saving.
- `train.py`:
  - Loads dataset, augments training data, trains TensorFlow model.
  - Evaluates with accuracy + confusion matrix and saves artifacts.
- `model.py`:
  - Defines neural network architecture and artifact load/predict helpers.
- `utils.py`:
  - Normalization, overlay drawing, FPS utility, gesture labels.
- `main.py`:
  - Real-time capture + MediaPipe landmarks + model inference overlay.
- `tests/test_pipeline.py`:
  - Verifies preprocessing and end-to-end training artifact generation.

## 10) Expected Outputs

1. Dataset CSV with 63 features + label.
2. Trained deep learning model and preprocessing artifacts.
3. Evaluation metrics JSON and confusion matrix image.
4. Live webcam prediction with confidence and FPS.

## 11) Troubleshooting

- **Webcam not opening**:
  - Check camera permissions.
  - Try `--camera-index 1`.
- **Model files missing**:
  - Run `train.py` before `main.py`.
- **MediaPipe import error**:
  - Reinstall deps: `pip install -r requirements.txt`
- **Low real-time accuracy**:
  - Collect more webcam samples per gesture with varied lighting/backgrounds.

## 12) Screenshots Section (for submission)

Add screenshots in this section after running locally:

1. `Dataset Creator` window showing captured landmarks and class count.
2. `Training` terminal output with final test accuracy.
3. `Confusion Matrix` image from `artifacts/confusion_matrix.png`.
4. `Main App` webcam window with predicted gesture, confidence, and FPS.

## 13) Final Working Workflow

1. Install dependencies.
2. Create dataset (`synthetic` or `webcam`).
3. Train model and generate artifacts.
4. Run real-time app with webcam.
5. Run tests for validation.

This repository is ready for direct university submission and live demo after local screenshot capture.
