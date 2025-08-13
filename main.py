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

def clean_gemini_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I | re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
        obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if arr_match:
            return json.loads(arr_match.group(1))
        if obj_match:
            return json.loads(obj_match.group(1))
    return None

def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(prompt).text

async def extract_url_and_questions(text):
    url_match = re.search(r'https?://\S+', text)
    url = url_match.group(0) if url_match else None
    questions = [q.strip() for q in re.findall(r'^\s*(?:\d+\.|-|\*)\s*(.+)', text, re.MULTILINE) if q.strip()]
    if url and questions:
        return {"url": url, "questions": questions}
    try:
        prompt = (
            "You are a helpful extractor. Extract from the text below:\n"
            "- 'url' (or null if none)\n"
            "- 'questions' as an array of strings.\n"
            "Return JSON with keys 'url' and 'questions'.\n"
            f"\nText:\n{text}"
        )
        resp = call_gemini(prompt)
        if not resp.strip():
            return {"url": None, "questions": []}
        parsed = json.loads(resp)
        if "url" in parsed and isinstance(parsed.get("questions"), list):
            return parsed
    except Exception:
        pass
    return {"url": None, "questions": []}

async def parse_file(name, content, ctype):
    try:
        if name.endswith(".json") or ctype == "application/json":
            return json.loads(content.decode(errors='ignore'))
        elif name.endswith(".txt"):
            try:
                return json.loads(content.decode(errors='ignore'))
            except Exception:
                return await extract_url_and_questions(content.decode(errors='ignore'))
        elif name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            preview = df.head().to_dict(orient='records')
            return {"csv_preview": preview, "row_count": len(df), "col_count": len(df.columns)}
        elif name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return await extract_url_and_questions(text)
        elif ctype.startswith("image/"):
            b64 = base64.b64encode(content).decode("utf-8")
            max_len = 130000  # approximately 100KB base64 size limit
            if len(b64) > max_len:
                b64 = b64[:max_len]
            return {"image_data_uri": f"data:{ctype};base64,{b64}"}
        else:
            return await extract_url_and_questions(content.decode(errors='ignore'))
    except Exception as e:
        return {"error": str(e)}

async def fetch_with_playwright(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')
        html = await page.content()
        await browser.close()
        return html

def extract_tables(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for tbl in soup.select("table.wikitable"):
        rows = tbl.select("tr")
        table_data = []
        for row in rows:
            cells = []
            for cell in row.find_all(["th","td"]):
                for t in cell.find_all(True):
                    t.decompose()
                cells.append(cell.get_text(strip=True))
            if cells:
                table_data.append(cells)
        if table_data:
            tables.append(table_data)
    return tables

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(['script','style','nav','aside','footer','header']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)

def build_prompt(url, tables, meta, questions, page_text=None, helpers=None):
    question_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = (
        "You are a precise data analyst. Use ONLY the provided tables and text.\n"
        "Do not use any external knowledge.\n"
        "Cleanse numbers before calculations, returning number types.\n"
        "Answer all questions exactly and only as requested.\n"
        "Return JSON matching the question order.\n"
        "For unanswerable questions, return null.\n"
        "All output must be raw JSON without markdown or explanations.\n"
    )
    if helpers:
        prompt += "\nHelper files included:\n"
        for fname, content in helpers.items():
            if isinstance(content, dict):
                preview = json.dumps(content)[:1000]
            else:
                preview = str(content)[:1000]
            prompt += f"- {fname}: {preview}\n"
    prompt += f"\nSource URL: {url}\nTables Metadata: {json.dumps(meta)}\nTables Content: {json.dumps(tables)}\nQuestions:\n{question_list}\n"
    if page_text:
        prompt += f"\nPage Text (trimmed): {page_text[:5000]}"
    return prompt

@app.get("/api")
async def health():
    return PlainTextResponse("API is healthy")

@app.post("/api")
async def analyze(request: Request):
    try:
        files = []
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            for _, f in form.multi_items():
                if isinstance(f, StarletteUploadFile):
                    files.append((f.filename.lower(), await f.read(), f.content_type))
        else:
            content = await request.body()
            files.append(("body.txt", content, "text/plain"))

        parsed_files = {}
        drivers = []

        # Parse and store all files
        for fname, content, ctype in files:
            parsed = await parse_file(fname, content, ctype)
            parsed_files[fname] = parsed
            if isinstance(parsed, dict) and "questions" in parsed and (("url" in parsed) or ("s3_path" in parsed)):
                drivers.append((fname, parsed))

        if not drivers:
            return JSONResponse({"results1": None, "files_parsed": parsed_files})

        results = {}
        for idx, (fname, driver) in enumerate(drivers, 1):
            if "s3_path" in driver and driver["s3_path"]:
                # DuckDB analytics: use LLM to generate SQL and run queries
                try:
                    from duckdb import connect
                    columns = ["court", "year", "date_of_registration", "decision_date", "disposal_nature",
                               "title", "description", "judge", "pdf_link", "cnr", "court_code", "raw_html", "bench"]
                    conn = connect(database=':memory:')
                    conn.execute("INSTALL httpfs; LOAD httpfs;")
                    conn.execute("INSTALL parquet; LOAD parquet;")
                    res = {}
                    for q in driver["questions"]:
                        sql_prompt = f"Create a DuckDB SQL query on columns {columns} to answer exactly: {q}\nUse {{DATA_PATH}} as source."
                        query = call_gemini(sql_prompt)
                        if not query:
                            res[q] = None
                            continue
                        query = query.strip().replace("{DATA_PATH}", f"'{driver['s3_path']}'")
                        try:
                            df = conn.execute(query).df()
                            if df.empty:
                                ans = None
                            elif df.shape == (1, 1):
                                ans = df.iloc[0, 0]
                            else:
                                ans = df.to_dict(orient='records')
                            res[q] = ans
                        except Exception as e:
                            res[q] = {"error": str(e), "query": query}
                    results[f"results{idx}"] = res
                except Exception as e:
                    results[f"results{idx}"] = {"error": str(e)}
            elif "url" in driver and driver["url"]:
                # Scrape with Playwright for dynamic content
                try:
                    html = await fetch_with_playwright(driver["url"])
                except Exception:
                    html = requests.get(driver["url"], timeout=30).text
                tables = extract_tables(html)
                meta = {
                    "num_tables": len(tables),
                    "tables": [{"table_index":i, "num_rows":len(table), "num_cols":max(len(row) for row in table)} for i,table in enumerate(tables)]
                }
                page_text = None if tables else extract_text(html)
                helpers = {k:v for k, v in parsed_files.items() if k != fname}
                prompt = build_prompt(driver["url"], tables, meta, driver["questions"], page_text, helpers)
                raw_ans = call_gemini(prompt)
                ans = clean_gemini_response(raw_ans)
                if ans is None:
                    ans = [None] * len(driver["questions"])
                results[f"results{idx}"] = ans
            else:
                results[f"results{idx}"] = None

        return JSONResponse(results)

    except Exception as e:
        return JSONResponse({
            "results1": None,
            "error": str(e),
            "trace": traceback.format_exc()
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=81)

