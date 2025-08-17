import os
import re
import io
import json
import base64
import traceback
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import google.generativeai as genai
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ------------- UPDATED: Dynamic base64 image sanitization function -------------
def sanitize_base64_images(obj, max_chars=100000):
    """
    Scan all dict keys for likely base64-encoded PNG/JPEG/GIF images
    and set oversized strings to "TOO_LARGE".
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Typical base64 headers for PNG, JPEG, GIF
            if (
                isinstance(v, str) and
                (
                    v.startswith("iVBORw0KGgo") or   # PNG
                    v.startswith("/9j/") or          # JPEG
                    v.startswith("R0lGODdh")         # GIF
                )
            ):
                if len(v) > max_chars:
                    obj[k] = "TOO_LARGE"
    return obj
# -------------------------------------------------------------------------------
def clean_gemini_response(text):
    text = text.strip()
    # 1. Extract content from any triple-backtick code block
    match = re.search(r"``````", text, re.DOTALL)
    candidate = None
    if match:
        candidate = match.group(1).strip()
        # Try to find the largest valid JSON object inside the block
        obj_match = re.search(r"(\{[\s\S]+\})", candidate)
        if obj_match:
            json_candidate = obj_match.group(1)
            try:
                obj = json.loads(json_candidate)
                obj = sanitize_base64_images(obj)  # <--- always sanitize base64 images
                return obj
            except Exception:
                pass # Ignore failure, fallback below
        # If direct JSON failed, try loading the whole block (if it appears to be JSON)
        try:
            obj = json.loads(candidate)
            obj = sanitize_base64_images(obj)
            return obj
        except Exception:
            pass
        # NEW: Try to extract JSON array from code block
        arr_match = re.search(r"(\[[\s\S]+\])", candidate)
        if arr_match:
            json_candidate = arr_match.group(1)
            try:
                arr = json.loads(json_candidate)
                return arr
            except Exception:
                pass

    # 2. Search whole text for any valid JSON object
    obj_match = re.search(r"(\{[\s\S]+\})", text)
    if obj_match:
        json_candidate = obj_match.group(1)
        try:
            obj = json.loads(json_candidate)
            obj = sanitize_base64_images(obj)
            return obj
        except Exception:
            pass
    # 3. NEW: Search for any valid JSON array in whole text
    arr_match = re.search(r"(\[[\s\S]+\])", text)
    if arr_match:
        json_candidate = arr_match.group(1)
        try:
            arr = json.loads(json_candidate)
            return arr
        except Exception:
            pass
     # === MINIMAL PATCH: Final fallback parses raw text as JSON ===
    try:
        obj = json.loads(text)
        obj = sanitize_base64_images(obj)
        return obj
    except Exception:
        pass
    # === END PATCH ===

    print("clean_gemini_response: Failed to extract JSON, raw LLM output:", text)
    if "Cannot answer with provided files" in text:
        return ["Cannot answer with provided files"]
    return None


def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(prompt).text

def build_flexible_llm_prompt(uploaded_files: dict) -> str:
    file_blocks = []
    if not uploaded_files:
        raise ValueError("No files uploaded")

    for fname, content in uploaded_files.items():
        ext = fname.lower().split('.')[-1]
        if ext == 'csv':
            try:
                content_str = content.decode('utf-8', errors='replace')
                content_preview = '\n'.join(content_str.splitlines()[:20])
                block = f"CSV file: {fname}\n---------------------\n{content_preview}\n---------------------"
            except Exception:
                block = f"CSV file: {fname} (Could not preview; unreadable)"
            file_blocks.append(block)
        elif ext == 'txt':
            try:
                content_str = content.decode('utf-8', errors='replace')
                content_preview = '\n'.join(content_str.splitlines()[:20])
                block = f"TEXT file: {fname} (may contain questions/instructions)\n---------------------\n{content_preview}\n---------------------"
            except Exception:
                block = f"TEXT file: {fname} (Could not preview)"
            file_blocks.append(block)
        elif ext in ['pdf', 'json']:
            block = f"{ext.upper()} file: {fname}\n(Non-previewable binary or structured file. Use only if you can read and extract content.)"
            file_blocks.append(block)
        else:
            block = f"Other file: {fname} (type: .{ext}) -- Not previewable here."
            file_blocks.append(block)

    files_section = "\n\n".join(file_blocks)
    prompt = f"""

You are a rigorous data analyst AI. The user has uploaded the following file(s):

{files_section}

Instructions:

1. Use ONLY the data/files shown above to answer the question; do not use outside knowledge.

2. If a file is a CSV, analyze it directly.

3. If a file is a text file and contains questions or instructions, follow them as specifically as possible.

4. If you have both a CSV and a questions file, answer all questions using ONLY the CSV data.

5. If you do NOT have enough information to answer any data question, reply 'Cannot answer with provided files'.

6. Always use only the information in the files above.

7. For any task that requires encoding PNG images to base64, if the resulting string exceeds 100,000 bytes, you MUST return "TOO_LARGE" in the output JSON for that field.

Output Contract:

- If any file gives a required schema or set of JSON keys, follow that exactly. If not, use your best judgment for a sensible, data-based answer. If the answer cannot be given, state as much.

- For any base64 PNG field, if the image is larger than 100,000 bytes, return the string "TOO_LARGE" instead.

- Output only what is required by the user's files and instructions.

- Your task is: Return ONLY a single pure JSON block as output, not python code, markdown, instructions, commentary, or explanation. Do not use code fences. Do not print anything before or after the JSON.

NEVER invent data not present in the uploaded files. NEVER guess.

Begin your answer now.

"""
    return prompt

@app.get("/api")
async def health():
    return PlainTextResponse("API is healthy")

@app.post("/api")
async def analyze(request: Request):
    try:
        uploaded_files = {}
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            for _, f in form.multi_items():
                if isinstance(f, StarletteUploadFile):
                    uploaded_files[f.filename] = await f.read()
                else:
                    content = await request.body()
                    uploaded_files["body.txt"] = content
        else:
            content = await request.body()
            uploaded_files["body.txt"] = content

        if not uploaded_files:
            return JSONResponse({"error": "No files uploaded"})

        prompt = build_flexible_llm_prompt(uploaded_files)
        print("==== LLM PROMPT ====")
        print(prompt)
        print("=====================")

        raw_ans = call_gemini(prompt)
        ans = clean_gemini_response(raw_ans)
        print("==== LLM RESPONSE ====")
        print(raw_ans)
        print("======================")

        # Unwrap "results1" if present at the top level
        if isinstance(ans, dict) and "results1" in ans:
            ans = ans["results1"]
        if ans is None:
            return JSONResponse({"answers": None, "error": "No valid answer from model"})
        return JSONResponse(ans)

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "trace": traceback.format_exc()
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=81)