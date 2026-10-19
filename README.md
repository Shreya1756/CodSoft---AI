# AI Resume Analyzer

Upload a PDF resume, paste a job description and get a match score, skill gap
and improvement suggestions. Uses TF-IDF and cosine similarity (scikit-learn).
Runs fully locally with no API key.

## Run (Windows)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Open http://127.0.0.1:5000 in your browser.

## Files
- app.py: Flask server and PDF reading
- analyzer.py: NLP logic (TF-IDF, cosine similarity, skill extraction)
- templates/: HTML pages
- static/: CSS and JavaScript
- samples/: sample job description and sample resume (PDF + text) for testing
