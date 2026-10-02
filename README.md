# AI Fake News and Misinformation Detector

## 1. Project introduction
This project is a beginner-friendly Flask web application that analyzes social media posts or article text and gives a quick AI-powered credibility score.

The app does not claim to know the absolute truth. Instead, it gives an automated assessment based on:
- NLP text classification
- Source verification against a trusted source list
- Basic article quality signals such as title, author, and publication date

It is designed for a hackathon demo and for learning how to build a practical ML + web application.

## 2. Features
- Paste article text or input a URL
- Fetch and parse webpage content using BeautifulSoup and requests
- Clean and preprocess content before model prediction
- Classify text as Potentially Reliable or Potentially Unreliable
- Estimate model confidence
- Verify whether the domain appears in a trusted source list
- Calculate a 0-100 credibility score using a transparent composite formula
- Save analysis history in SQLite
- Show the result and explanation in a clean frontend

## 3. Technology stack
- Python
- Flask
- HTML
- CSS
- JavaScript
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- BeautifulSoup
- requests
- SQLite
- joblib

## 4. Project architecture
The project is split into small modules so each part is easier to explain and maintain.

- app.py: Flask routes and main logic
- preprocess.py: text cleaning and feature preparation
- train_model.py: dataset loading, model training, and metric reporting
- source_verification.py: URL validation, webpage extraction, and source checks
- database.py: SQLite history storage

## 5. Folder structure
```text
FakeNewsDetector/
├── app.py
├── train_model.py
├── preprocess.py
├── source_verification.py
├── database.py
├── requirements.txt
├── README.md
├── dataset/
│   └── fake_news.csv
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
├── static/
│   ├── style.css
│   └── script.js
├── data/
│   └── trusted_sources.json
├── tests/
│   ├── test_preprocess.py
│   └── test_source_verification.py
└── analysis_history.db
```

## 6. Installation steps
Open a terminal in the project folder and create a virtual environment:

```bash
cd FakeNewsDetector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Virtual environment setup
On Windows PowerShell:

```powershell
cd FakeNewsDetector
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows Command Prompt:

```cmd
cd FakeNewsDetector
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 8. Package installation
Use:

```bash
pip install -r requirements.txt
```

This installs Flask, pandas, scikit-learn, requests, BeautifulSoup, joblib, and pytest.

## 9. Dataset setup
Place a public fake-news dataset inside:

```text
FakeNewsDetector/dataset/fake_news.csv
```

The CSV should include a text/article column and a label/class column, such as:

```csv
text,label
"This article says the moon is made of cheese.","unreliable"
"The central bank announced a new policy update.","reliable"
```

If the dataset is missing or empty, the training script will print a clear setup message instead of pretending a model was created.

## 10. Model training command
Run:

```bash
python train_model.py
```

This script:
- reads the CSV
- cleans the text
- splits data into train and test sets
- creates TF-IDF features
- trains Logistic Regression
- prints accuracy, precision, recall, F1-score, and confusion matrix
- saves the model and vectorizer

## 11. Flask application start command
Start the web application with:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## 12. How to use the application
1. Open the homepage.
2. Choose either Paste Text or Analyze URL.
3. Paste article text or enter a URL.
4. Press Analyze.
5. Read the result card, score breakdown, and explanation.
6. Use the History page to review previous results.

## 13. How credibility scoring works
The app uses a composite score that is intentionally transparent and project-defined.

The score is built from three parts:
- NLP model result: 50%
- Source verification: 25%
- Article/source quality signals: 25%

Important:
- This is not a probability that a claim is true.
- The app clearly says it is an AI/NLP assessment.
- Users should still verify important information with independent sources.

## 14. Limitations
- The model is only as good as the training dataset.
- A known publisher does not guarantee every article is fully true.
- URL extraction can fail on some websites.
- Short text can produce less reliable results.
- This tool is for educational and demo use, not as a legal or fact-checking authority.

## 15. Future improvements
- Add a larger real-world fake-news dataset
- Include more source risk categories
- Add explainable AI output for each classification decision
- Add user authentication for saved history
- Improve article extraction for difficult websites
- Add stronger moderation of text inputs

## 16. Hackathon presentation explanation
A short script you can say in a presentation:

> This project builds an AI-based fake-news detector for social media and online articles. It takes either pasted text or a URL, cleans the content, and runs a machine learning model trained with TF-IDF and Logistic Regression. It also checks whether the domain is a known publisher, calculates a credibility score, and explains the result in simple language. The main idea is to help users assess content cautiously, not to claim absolute truth. The app clearly states that results are automated and should be independently verified.

## 17. Basic testing strategy
The project includes basic tests for:
- clean text preprocessing
- URL validation
- known and unknown source checks

Run tests with:

```bash
pytest
```

If you want to add more tests, keep them focused on real behavior and avoid fake data.
