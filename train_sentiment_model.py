from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "comments.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "comment_sentiment_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "sentiment_metrics.json"

ALLOWED_LABELS = ["Positive", "Neutral", "Negative"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["manual_sentiment"].isin(ALLOWED_LABELS)].copy()
    if df.empty:
        raise ValueError("No labeled sentiment rows found in comments.csv")

    df["text"] = (
        df["normalized_text"]
        .fillna(df["comment_text"])
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df[["comment_id", "text", "manual_sentiment"]].dropna(subset=["text"]).reset_index(drop=True)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        labels=ALLOWED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_test, predictions, labels=ALLOWED_LABELS).tolist()
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": {"labels": ALLOWED_LABELS, "matrix": conf_matrix},
    }


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    X = dataset["text"]
    y = dataset["manual_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Dataset summary:")
    print(f"- Rows used for training: {len(dataset)}")
    print(f"- Sentiment classes: {dataset['manual_sentiment'].value_counts().to_dict()}")
    print("\nModel performance:")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- Macro F1: {metrics['classification_report']['macro avg']['f1-score']:.4f}")
    print("\nSaved artifacts:")
    print(f"- Model: {MODEL_PATH}")
    print(f"- Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
