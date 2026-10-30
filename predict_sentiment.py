from __future__ import annotations

import sys
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "comment_sentiment_model.joblib"


def ensure_model() -> None:
    if not MODEL_PATH.exists():
        print("Model not found. Training a new one...")
        import subprocess

        subprocess.run([sys.executable, str(BASE_DIR / "train_sentiment_model.py")], check=True)


def predict_text(text: str) -> str:
    ensure_model()
    model = joblib.load(MODEL_PATH)
    return model.predict([text])[0]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: py predict_sentiment.py \"your comment text\"")
        raise SystemExit(1)

    comment = " ".join(sys.argv[1:])
    prediction = predict_text(comment)
    print(f"Predicted sentiment: {prediction}")


if __name__ == "__main__":
    main()
