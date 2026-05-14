import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_PATH = "model.pkl"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df["clean_text"] = df["comment_text"].astype(str).apply(preprocess)
    df["label"] = (df[LABEL_COLUMNS].sum(axis=1) > 0).astype(int)
    return df["clean_text"], df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=5,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        ))
    ])


def train(csv_path: str) -> dict:
    print("Loading data...")
    texts, df = load_data(csv_path)
    y_binary = df["label"]
    y_multi = df[LABEL_COLUMNS]

    X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
        texts, y_binary, y_multi,
        test_size=0.2, random_state=42, stratify=y_binary
    )

    print("Training binary classifier...")
    binary_model = build_pipeline()
    binary_model.fit(X_train, yb_train)

    print("Training per-label classifiers...")
    label_models = {}
    for col in LABEL_COLUMNS:
        m = build_pipeline()
        m.fit(X_train, ym_train[col])
        label_models[col] = m

    print("\nPer-Label Classification Reports:")
    for col, m in label_models.items():
        ym_pred = m.predict(X_test)
        print(f"\n--- {col} ---")
        print(classification_report(ym_test[col], ym_pred, zero_division=0))

    yb_pred = binary_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(yb_test, yb_pred),
        "precision": precision_score(yb_test, yb_pred, zero_division=0),
        "recall": recall_score(yb_test, yb_pred, zero_division=0),
        "f1": f1_score(yb_test, yb_pred, zero_division=0),
    }

    print("\nBinary Classification Report:")
    print(classification_report(yb_test, yb_pred, target_names=["Non-Toxic", "Toxic"]))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"binary": binary_model, "labels": label_models}, f)

    print(f"Model saved to {MODEL_PATH}")
    _plot_confusion_matrix(yb_test, yb_pred)
    _plot_metrics(metrics)

    return metrics


def load_model() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(text: str, models: dict = None) -> dict:
    if models is None:
        models = load_model()

    clean = preprocess(text)
    binary_model = models["binary"]
    label_models = models["labels"]

    is_toxic = bool(binary_model.predict([clean])[0])
    toxic_prob = binary_model.predict_proba([clean])[0][1]

    label_probs = {}
    for col, m in label_models.items():
        prob = m.predict_proba([clean])[0]
        classes = list(m.classes_)
        positive_prob = prob[classes.index(1)] if 1 in classes else 0.0
        label_probs[col] = round(float(positive_prob) * 100, 2)

    return {
        "text": text,
        "toxic": is_toxic,
        "toxicity_score": round(float(toxic_prob) * 100, 2),
        "label_scores": label_probs,
        "detected_labels": [k for k, v in label_probs.items() if v >= 50.0],
    }


def format_result(result: dict) -> str:
    lines = [
        f"\nInput : {result['text']}",
        f"Toxic  : {'YES' if result['toxic'] else 'NO'}",
        f"Score  : {result['toxicity_score']}%",
        "\nCategory Scores:",
    ]
    for label, score in result["label_scores"].items():
        lines.append(f"  {label:<15} {score}%")
    if result["detected_labels"]:
        lines.append(f"\nDetected: {', '.join(result['detected_labels'])}")
    return "\n".join(lines)


def _plot_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Toxic", "Toxic"],
                yticklabels=["Non-Toxic", "Toxic"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


def _plot_metrics(metrics: dict) -> None:
    names = list(metrics.keys())
    values = [metrics[k] for k in names]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
                 f"{val:.3f}", ha="center", va="top", color="white", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.title("Model Evaluation Metrics")
    plt.tight_layout()
    plt.savefig("metrics.png", dpi=150)
    plt.close()
    print("Saved metrics.png")


def interactive_mode() -> None:
    print("\nLoading model...")
    models = load_model()
    print("Ready. Type 'quit' to exit.\n")
    while True:
        text = input("Enter text: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        result = predict(text, models)
        print(format_result(result))
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cyberbullying_detector.py train <path_to_train.csv>")
        print("  python cyberbullying_detector.py predict")
        print("  python cyberbullying_detector.py predict \"Your text here\"")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "train":
        if len(sys.argv) < 3:
            print("Provide path to train.csv")
            sys.exit(1)
        train(sys.argv[2])

    elif command == "predict":
        if len(sys.argv) >= 3:
            text_input = " ".join(sys.argv[2:])
            models = load_model()
            result = predict(text_input, models)
            print(format_result(result))
        else:
            interactive_mode()

    else:
        print(f"Unknown command: {command}")
