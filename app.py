from pathlib import Path

import joblib
from flask import Flask, render_template, request

from database import get_history, init_db, save_analysis
from preprocess import clean_text
from source_verification import extract_article_metadata, validate_url, verify_source

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

app = Flask(__name__)
init_db()


def load_model_artifacts():
    """Load the trained model and vectorizer if they exist."""
    if (
        not MODEL_PATH.exists()
        or not VECTORIZER_PATH.exists()
        or MODEL_PATH.stat().st_size == 0
        or VECTORIZER_PATH.stat().st_size == 0
    ):
        raise FileNotFoundError(
            "The model files are missing or not trained yet. Run 'python train_model.py' after placing a dataset in dataset/fake_news.csv."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_label(prediction_value):
    """Map the model class to a friendlier label for the user interface."""
    return "Potentially Unreliable" if int(prediction_value) == 1 else "Potentially Reliable"


def calculate_source_score(source_status):
    """Assign a score for the source verification component."""
    score_map = {
        "Known publisher": 25,
        "Source not verified": 12,
        "Unable to verify source": 5,
    }
    return score_map.get(source_status, 0)


def calculate_quality_score(title, author, publication_date, article_text):
    """Score basic article metadata quality signals."""
    quality_points = 0

    if title:
        quality_points += 8
    if author:
        quality_points += 8
    if publication_date:
        quality_points += 9
    if article_text and len(article_text) > 200:
        quality_points += 10
    if article_text and len(article_text) > 1000:
        quality_points += 10

    return min(25, quality_points)


def build_explanation(prediction_label, source_status, metadata):
    """Create a human-readable explanation from the available signals."""
    reasons = []

    if prediction_label == "Potentially Unreliable":
        reasons.append("The NLP model classified the text as potentially unreliable.")
    else:
        reasons.append("The NLP model classified the text as potentially reliable.")

    if source_status == "Known publisher":
        reasons.append("The source matched a known publisher in the configured source list.")
    elif source_status == "Source not verified":
        reasons.append("The source could not be verified in the configured source database.")
    else:
        reasons.append("The source could not be verified, so the assessment remains cautious.")

    if not metadata.get("title"):
        reasons.append("Article metadata was incomplete because no title was detected.")
    if not metadata.get("author"):
        reasons.append("Author information was not found for this article.")
    if not metadata.get("publication_date"):
        reasons.append("Publication date information was not found in the page metadata.")

    if not metadata.get("text") or len(metadata.get("text", "")) < 150:
        reasons.append("The article text was short, which may reduce confidence in the assessment.")

    return " ".join(reasons)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/history")
def history():
    entries = get_history(limit=20)
    return render_template("history.html", history=entries)


@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = (request.form.get("text") or "").strip()
    url_input = (request.form.get("url") or "").strip()

    if not text_input and not url_input:
        return render_template("index.html", error="Please enter some text or a valid article URL."), 400

    metadata = {
        "title": "",
        "author": "",
        "publication_date": "",
        "text": "",
        "domain": "",
    }

    if url_input:
        if not validate_url(url_input):
            return render_template("index.html", error="The URL is invalid. Please use a full http:// or https:// URL."), 400
        try:
            metadata = extract_article_metadata(url_input)
        except ValueError as exc:
            return render_template("index.html", error=str(exc)), 400

        if not metadata.get("text"):
            return render_template("index.html", error="The webpage was accessed but no readable article text could be extracted."), 400

        source_check = verify_source(metadata.get("domain"))
        content = metadata["text"]
        input_type = "url"
        input_value = url_input
    else:
        content = text_input
        input_type = "text"
        input_value = text_input[:200]
        source_check = {
            "status": "Source not verified",
            "publisher": "Unknown",
            "domain": "Manual text input",
            "category": "unknown",
        }

    cleaned_content = clean_text(content)
    if not cleaned_content:
        return render_template("index.html", error="The provided text is empty after cleaning. Please enter readable content."), 400

    try:
        model, vectorizer = load_model_artifacts()
        features = vectorizer.transform([cleaned_content])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        class_scores = list(zip(model.classes_, probabilities))
        predicted_class, predicted_probability = max(class_scores, key=lambda item: item[1])
        model_confidence = float(predicted_probability * 100)
        prediction_label = predict_label(predicted_class)
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc)), 500
    except Exception as exc:
        return render_template("index.html", error=f"Model prediction failed: {exc}"), 500

    source_status = source_check.get("status", "Unable to verify source")
    source_score = calculate_source_score(source_status)
    quality_score = calculate_quality_score(
        metadata.get("title", ""),
        metadata.get("author", ""),
        metadata.get("publication_date", ""),
        metadata.get("text", content),
    )

    nlp_score = round(model_confidence * 0.5, 2)
    total_score = min(100, round(nlp_score + source_score + quality_score))

    result = {
        "credibility_score": total_score,
        "prediction": prediction_label,
        "model_confidence": round(model_confidence, 2),
        "source_status": source_status,
        "source_publisher": source_check.get("publisher", "Unknown"),
        "domain": source_check.get("domain", metadata.get("domain", "Manual text input")),
        "title": metadata.get("title") or "No title detected",
        "author": metadata.get("author") or "Not available",
        "publication_date": metadata.get("publication_date") or "Not available",
        "score_breakdown": {
            "nlp": f"{round(nlp_score, 2)}/50",
            "source": f"{source_score}/25",
            "quality": f"{quality_score}/25",
        },
        "explanation": build_explanation(prediction_label, source_status, metadata),
        "disclaimer": "This tool provides an automated NLP-based assessment. It does not prove whether a claim is true or false. Always verify important information using reliable independent sources.",
    }

    save_analysis(
        {
            "input_type": input_type,
            "input_value": input_value,
            "title": result["title"],
            "domain": result["domain"],
            "prediction": result["prediction"],
            "model_confidence": result["model_confidence"],
            "source_status": result["source_status"],
            "credibility_score": result["credibility_score"],
        }
    )

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
