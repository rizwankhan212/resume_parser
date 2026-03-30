import os
import json
import re
import pdfplumber
import docx

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types

# ==============================
# 🔑 API KEY
# ==============================
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)

# ==============================
# 🚀 FASTAPI APP
# ==============================
app = FastAPI(
    title="Resume Parser API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🏠 HEALTH CHECK
# ==============================
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# ==============================
# 📄 TEXT EXTRACTION
# ==============================
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")


# ==============================
# 🧠 TEXT CLEANING
# ==============================
def clean_text(text):
    return text.replace("\n\n", "\n").strip()


def limit_text(text, max_chars=8000):
    return text[:max_chars]


# ==============================
# 🤖 PROMPT
# ==============================
system_prompt = (
    "You are a resume parser.\n\n"
    "Extract structured data and return ONLY valid JSON.\n\n"
    "Rules:\n"
    "- No explanation\n"
    "- No extra text\n"
    "- No markdown\n"
    "- No trailing commas\n"
    "- All strings must be in double quotes\n\n"
    "Output format:\n"
    "{\n"
    '  "full_name": "",\n'
    '  "email": "",\n'
    '  "github": "",\n'
    '  "linkedin": "",\n'
    '  "employment_details": [],\n'
    '  "technical_skills": [],\n'
    '  "soft_skills": []\n'
    "}"
)

# ==============================
# 🛡️ SAFE JSON PARSER
# ==============================
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group()
            json_str = json_str.replace("\n", " ").replace("\t", " ")
            return json.loads(json_str)

    raise ValueError("Invalid JSON from model")


# ==============================
# 🔁 RETRY LOGIC
# ==============================
def ats_extractor_with_retry(text):
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )

            result = response.text

            return safe_json_parse(result)

        except Exception as e:
            print(f"Attempt {attempt+1} failed:", e)

    raise ValueError("Model failed to return valid JSON")


# ==============================
# 📤 UPLOAD ENDPOINT
# ==============================
@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):

    if not file.filename.endswith((".pdf", ".docx")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF and DOCX files supported"}
        )

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        raw_text = extract_text(file_path)
        cleaned_text = clean_text(raw_text)
        limited_text = limit_text(cleaned_text)

        if not limited_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text"}
            )

        data = ats_extractor_with_retry(limited_text)

        return {
            "success": True,
            "filename": file.filename,
            "data": data
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)