import os
import json
import pdfplumber
import docx
import re

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types

# ==============================
# 🔑 API KEY (ENV VARIABLE)
# ==============================
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=api_key)

# ==============================
# 🚀 FASTAPI APP
# ==============================
app = FastAPI(
    title="Resume Parser API",
    description="Upload resume (PDF/DOCX) and get structured JSON data",
    version="1.0"
)

# ==============================
# 🌐 CORS (for frontend)
# ==============================
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
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")


def clean_text(text):
    return text.replace("\n\n", "\n").strip()

# ==============================
# 🤖 GEMINI PARSER
# ==============================
def ats_extractor(resume_data):
    system_prompt = (
    "Extract resume details and return ONLY valid JSON.\n\n"
    "Format strictly like this:\n"
    "{\n"
    '  "full_name": "",\n'
    '  "email": "",\n'
    '  "github": "",\n'
    '  "linkedin": "",\n'
    '  "employment_details": [\n'
    '    {"title": "", "company": "", "duration": ""}\n'
    "  ],\n"
    '  "technical_skills": [],\n'
    '  "soft_skills": []\n'
    "}\n\n"
    "No explanation. No extra text. Only JSON."
)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=resume_data,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=1500,
            response_mime_type="application/json"
        )
    )

    return response.text

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        # Extract JSON part using regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("Invalid JSON from model")

# ==============================
# 📤 UPLOAD ENDPOINT
# ==============================
@app.post("/upload", summary="Upload Resume")
async def upload_resume(file: UploadFile = File(...)):

    # Validate file type
    if not file.filename.endswith((".pdf", ".docx")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF and DOCX files are supported"}
        )

    # Save file temporarily
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extract text
        raw_text = extract_text(file_path)
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text (maybe scanned PDF?)"}
            )

        # Parse with Gemini
        result = ats_extractor(cleaned_text)
        data = safe_json_parse(result)

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
        # Delete temp file
        if os.path.exists(file_path):
            os.remove(file_path)