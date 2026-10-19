"""
NLP logic for the AI Resume Analyzer.
Runs 100% locally: no API key needed.

Techniques used:
  1. Text preprocessing (lowercasing, cleaning, stop-word removal)
  2. TF-IDF vectorization
  3. Cosine similarity
  4. Dictionary-based skill extraction
  5. Section extraction (education / experience / projects)
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1. SKILL DATABASE  (skill name -> list of ways it can be written)
# ----------------------------------------------------------------------
SKILLS = {
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "ecmascript"],
    "TypeScript": ["typescript"],
    "C": ["c programming", "c language"],
    "C++": ["c++"],
    "C#": ["c#"],
    "Go": ["golang"],
    "R": ["r programming", "rstudio"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb"],
    "Firebase": ["firebase"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Bootstrap": ["bootstrap"],
    "Tailwind": ["tailwind", "tailwindcss"],
    "React": ["react", "reactjs", "react.js"],
    "Angular": ["angular", "angularjs"],
    "Vue": ["vue", "vuejs", "vue.js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express": ["express.js", "expressjs"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "springboot"],
    "REST API": ["rest api", "rest apis", "restful api", "restful apis", "restful"],
    "Streamlit": ["streamlit"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Linux": ["linux"],
    "CI/CD": ["ci/cd", "cicd"],
    "Jenkins": ["jenkins"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "Generative AI": ["generative ai", "genai"],
    "LLM": ["llm", "llms", "large language models"],
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis", "data analytics", "data analyst"],
    "Statistics": ["statistics", "statistical analysis"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Keras": ["keras"],
    "OpenCV": ["opencv"],
    "Matplotlib": ["matplotlib"],
    "Spark": ["spark", "pyspark"],
    "Hadoop": ["hadoop"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["ms excel", "microsoft excel", "advanced excel"],
    "Data Structures": ["data structures", "dsa"],
    "Algorithms": ["algorithms"],
    "OOP": ["oop", "oops", "object oriented", "object-oriented"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Jira": ["jira"],
    "Postman": ["postman"],
    "Selenium": ["selenium"],
    "Testing": ["unit testing", "software testing", "test automation"],
    "Figma": ["figma"],
}

# Pre-compile patterns. The look-behind / look-ahead make sure that
# "java" does NOT match inside "javascript" and "sql" does NOT match "mysql".
SKILL_PATTERNS = {
    name: [
        re.compile(r"(?<![a-z0-9+#])" + re.escape(alias) + r"(?![a-z0-9+#])")
        for alias in aliases
    ]
    for name, aliases in SKILLS.items()
}

# ----------------------------------------------------------------------
# 2. SECTION HEADINGS (used to find Education / Experience / Projects)
# ----------------------------------------------------------------------
HEADINGS = {
    "education": ["education", "academic background", "academics",
                  "qualifications", "academic qualifications"],
    "experience": ["experience", "work experience", "professional experience",
                   "internship", "internships", "employment", "work history",
                   "employment history"],
    "projects": ["projects", "academic projects", "personal projects",
                 "project work", "key projects"],
    "skills": ["skills", "technical skills", "key skills", "core competencies",
               "skills & tools"],
    "other": ["certifications", "achievements", "awards", "summary", "objective",
              "profile", "about me", "career objective", "professional summary",
              "interests", "hobbies", "languages", "declaration", "references",
              "extracurricular activities", "publications", "contact"],
}
HEADING_MAP = {alias: section for section, aliases in HEADINGS.items() for alias in aliases}

BULLET_RE = re.compile(r"^[\-\*\u2022\u25aa\u25cf\u00b7\s]+")
DEGREE_RE = re.compile(
    r"(b\.?tech|m\.?tech|bachelor|master|mba|ph\.?d|diploma|b\.?sc|m\.?sc|"
    r"b\.?e\.|university|college|institute)",
    re.IGNORECASE,
)

# Extra words that carry no useful meaning in a job description
STOP_WORDS = list(
    ENGLISH_STOP_WORDS.union({
        "experience", "work", "working", "years", "year", "team", "teams", "ability",
        "strong", "good", "knowledge", "skills", "skill", "required", "requirements",
        "responsibilities", "preferred", "looking", "candidate", "role", "job",
        "position", "company", "including", "etc", "using", "use", "like", "ensure",
        "understanding", "excellent", "plus", "basic", "familiarity", "exposure",
        "help", "build", "product",
    })
)

ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "created", "led", "improved",
    "optimized", "managed", "analyzed", "deployed", "automated", "engineered",
    "trained", "integrated",
]


# ----------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------------------------
def preprocess(text):
    """Basic NLP cleaning: lowercase, remove links/emails/numbers/symbols."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_skills(text):
    """Return the list of known skills that appear in the text."""
    t = re.sub(r"\s+", " ", text.lower())
    return [name for name, patterns in SKILL_PATTERNS.items()
            if any(p.search(t) for p in patterns)]


def extract_sections(text):
    """Split the resume into Education / Experience / Projects lines."""
    sections = {"education": [], "experience": [], "projects": []}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = " ".join(re.sub(r"[^a-z& ]", " ", line.lower()).split())
        if len(line) <= 40 and key in HEADING_MAP:
            current = HEADING_MAP[key]
            continue
        if current in sections:
            cleaned = BULLET_RE.sub("", line).strip()
            if cleaned:
                sections[current].append(cleaned[:200])

    # Fallback: if no Education heading was found, look for degree words
    if not sections["education"]:
        for raw in text.splitlines():
            if DEGREE_RE.search(raw):
                sections["education"].append(BULLET_RE.sub("", raw).strip()[:200])

    return {key: value[:8] for key, value in sections.items()}


def tfidf_similarity(resume_text, jd_text, missing_skills):
    """
    Convert both texts to TF-IDF vectors and compare them with cosine similarity.
    Also returns important job-description words that are absent from the resume.
    """
    r, j = preprocess(resume_text), preprocess(jd_text)
    if not r or not j:
        return 0.0, []
    try:
        vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
        matrix = vectorizer.fit_transform([r, j])
    except ValueError:
        return 0.0, []

    similarity = float(cosine_similarity(matrix[0], matrix[1])[0][0])

    names = vectorizer.get_feature_names_out()
    resume_vec = matrix[0].toarray()[0]
    jd_vec = matrix[1].toarray()[0]
    missing_text = " ".join(missing_skills).lower()

    keywords = []
    for i in jd_vec.argsort()[::-1]:
        if jd_vec[i] <= 0:
            break
        word = names[i]
        if resume_vec[i] == 0 and word not in missing_text:
            keywords.append(word)
        if len(keywords) == 8:
            break
    return similarity, keywords


def build_suggestions(resume_text, missing, keywords, sections, score):
    tips = []
    lower = resume_text.lower()
    words = len(resume_text.split())

    if missing:
        tips.append("Add the missing skills you genuinely have (" + ", ".join(missing[:6])
                    + ") to your Skills section and mention them in project or work bullets.")
        tips.append("If you do not know some of them yet, list a small project or course "
                    "that shows you are learning them.")
    if keywords:
        tips.append("Naturally use important words from the job description: "
                    + ", ".join(keywords[:6]) + ".")
    if not sections["projects"]:
        tips.append("Add a Projects section with 2-3 projects. Name the tools used and the result.")
    if not sections["experience"]:
        tips.append("Add an Experience or Internship section, even small internships, "
                    "freelance work or volunteering.")
    if not sections["education"]:
        tips.append("Add a clear Education section with degree, college and year.")
    if not re.search(r"\d+\s*%", resume_text):
        tips.append("Add measurable results (for example 'improved accuracy by 15%' "
                    "or 'handled 500+ records').")
    used_verbs = sum(1 for v in ACTION_VERBS if v in lower)
    if used_verbs < 3:
        tips.append("Start bullet points with action verbs such as Developed, Built, "
                    "Designed, Implemented or Optimized.")
    if not re.search(r"\S+@\S+", resume_text):
        tips.append("Add a professional email address at the top of your resume.")
    if "linkedin" not in lower and "github" not in lower:
        tips.append("Add your LinkedIn and/or GitHub links.")
    if words < 200:
        tips.append("Your resume looks short (" + str(words) + " words). Add more detail "
                    "about projects and skills.")
    elif words > 900:
        tips.append("Your resume is long (" + str(words) + " words). Aim for one page "
                    "for an internship.")
    if score >= 75 and len(tips) < 2:
        tips.append("Great match! Proofread once more and tailor your summary to this role.")
    if not tips:
        tips.append("Your resume looks well structured. Keep tailoring it for every job.")
    return tips


def build_summary(resume_text, detected, sections, years):
    parts = []
    if detected:
        parts.append("The candidate shows skills in " + str(len(detected))
                     + " technical areas, including " + ", ".join(detected[:6]) + ".")
    else:
        parts.append("No common technical skills were detected in this resume.")

    if sections["education"]:
        parts.append("Education: " + sections["education"][0].rstrip(".") + ".")
    if years:
        parts.append("The resume mentions about " + years + " year(s) of experience.")
    elif sections["experience"]:
        parts.append("The resume includes an experience/internship section.")
    else:
        parts.append("No experience section was found.")
    if sections["projects"]:
        parts.append("It also includes a projects section.")
    parts.append("Resume length: about " + str(len(resume_text.split())) + " words.")
    return " ".join(parts)


# ----------------------------------------------------------------------
# 4. MAIN FUNCTION
# ----------------------------------------------------------------------
def analyze(resume_text, jd_text):
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)

    matching = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]

    # Score part 1: how many of the required skills are present
    skill_score = len(matching) / len(jd_skills) if jd_skills else None

    # Score part 2: TF-IDF + cosine similarity of the full texts.
    # Cosine of 0.5 is already a very strong match for long texts, so we
    # scale it: 0.5 or more counts as 100% text similarity.
    cosine, keywords = tfidf_similarity(resume_text, jd_text, missing)
    text_score = min(cosine / 0.5, 1.0)

    if skill_score is None:
        final = text_score
    else:
        final = 0.6 * skill_score + 0.4 * text_score
    score = int(round(final * 100))

    sections = extract_sections(resume_text)
    years_found = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", resume_text.lower())
    years = max(years_found, key=float) if years_found else None

    if score >= 75:
        verdict = "Strong match"
    elif score >= 50:
        verdict = "Moderate match"
    else:
        verdict = "Low match"

    return {
        "score": score,
        "verdict": verdict,
        "skill_score": None if skill_score is None else int(round(skill_score * 100)),
        "text_score": int(round(text_score * 100)),
        "matching": matching,
        "missing": missing,
        "detected": resume_skills,
        "keywords": keywords,
        "education": sections["education"],
        "experience": sections["experience"],
        "projects": sections["projects"],
        "years": years,
        "summary": build_summary(resume_text, resume_skills, sections, years),
        "suggestions": build_suggestions(resume_text, missing, keywords, sections, score),
    }
