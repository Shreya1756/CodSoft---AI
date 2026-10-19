"""Flask web server for the AI Resume Analyzer."""
import os
import uuid

from flask import Flask, render_template, request
from pypdf import PdfReader
from werkzeug.utils import secure_filename

from analyzer import analyze

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit


def extract_text_from_pdf(path):
    """Read every page of the PDF and return all text as one string."""
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt("")
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


@app.route("/")
def index():
    return render_template("index.html", error=None, jd="")


@app.route("/analyze", methods=["POST"])
def analyze_resume():
    jd = request.form.get("job_description", "").strip()
    file = request.files.get("resume")

    # ---- Validate input ----
    if file is None or file.filename == "":
        return render_template("index.html", error="Please upload your resume as a PDF.", jd=jd)
    if not file.filename.lower().endswith(".pdf"):
        return render_template("index.html", error="Only PDF files are supported.", jd=jd)
    if len(jd) < 30:
        return render_template(
            "index.html",
            error="Please paste a longer job description (at least a few lines).",
            jd=jd,
        )

    # ---- Save, read, delete ----
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)

    try:
        resume_text = extract_text_from_pdf(save_path)
    except Exception:
        return render_template(
            "index.html",
            error="Could not read this PDF. Try another file.",
            jd=jd,
        )
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    if len(resume_text.strip()) < 50:
        return render_template(
            "index.html",
            error="No readable text found. The PDF may be a scanned image. "
                  "Please use a PDF with selectable text.",
            jd=jd,
        )

    result = analyze(resume_text, jd)
    return render_template("result.html", r=result)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="File is too large (max 5 MB).", jd=""), 413


if __name__ == "__main__":
    app.run(debug=True)
