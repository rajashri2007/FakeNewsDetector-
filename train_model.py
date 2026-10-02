import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "fake_news.csv"
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

TEXT_COLUMN_CANDIDATES = ["text", "content", "article", "body", "headline", "title"]
LABEL_COLUMN_CANDIDATES = ["label", "class", "target", "category", "verdict"]


def detect_column(columns, candidates):
    """Return the first matching column name in a list of possible names."""
    lower_columns = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_columns:
            return lower_columns[candidate.lower()]
    return None


def map_label(value):
    """Map common labels into two classes: 0 = reliable, 1 = unreliable."""
    if value is None:
        return 1

    label = str(value).strip().lower()

    reliable_words = {"reliable", "real", "true", "trustworthy", "credible", "not fake", "safe"}
    unreliable_words = {
        "unreliable",
        "fake",
        "fake news",
        "false",
        "misinformation",
        "misleading",
        "rumor",
        "not reliable",
        "hoax",
    }

    if label in reliable_words or label in {"0", "0.0"}:
        return 0
    if label in unreliable_words or label in {"1", "1.0"}:
        return 1

    if "reliable" in label or "true" in label or "real" in label:
        return 0
    if "unreliable" in label or "fake" in label or "misinfo" in label or "false" in label:
        return 1

    raise ValueError(f"Unsupported label value found: '{value}'. Use reliable/unreliable or similar values.")


def train_model():
    """Load the CSV dataset, train a model, and save it to disk."""
    if not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0:
        print("Dataset setup required: place a public fake-news CSV file at dataset/fake_news.csv.")
        print("Expected columns: a text/article column and a label/class column such as text,label.")
        return False

    try:
        data = pd.read_csv(DATASET_PATH)
    except Exception as exc:
        print(f"Could not read the dataset: {exc}")
        return False

    if data.empty:
        print("The dataset is present but has no rows to train with.")
        return False

    text_column = detect_column(data.columns, TEXT_COLUMN_CANDIDATES)
    label_column = detect_column(data.columns, LABEL_COLUMN_CANDIDATES)

    if text_column is None or label_column is None:
        print("The dataset does not include the expected text and label columns.")
        print(f"Found columns: {list(data.columns)}")
        print("Please make sure the CSV includes a text/article column and a label/class column.")
        return False

    selected = data[[text_column, label_column]].dropna().copy()

    if selected.empty:
        print("After removing empty rows, the dataset is blank.")
        return False

    selected["cleaned_text"] = selected[text_column].apply(clean_text)
    selected = selected[selected["cleaned_text"].str.len() > 0]

    if selected.empty:
        print("No usable text remained after cleaning. Please check your dataset contents.")
        return False

    try:
        selected["label_code"] = selected[label_column].apply(map_label)
    except ValueError as exc:
        print(str(exc))
        return False

    X = selected["cleaned_text"]
    y = selected["label_code"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model training completed successfully.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    return True


if __name__ == "__main__":
    try:
        success = train_model()
        if not success:
            sys.exit(1)
    except Exception as exc:
        print(f"Training failed: {exc}")
        sys.exit(1)
