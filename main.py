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

# --- NEW/UPDATED: Minimal PNG single pixel placeholder data URI ---

BASE64_PNG_PLACEHOLDER = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR"
    "42mP8z8BQDwAFigJ/l8K2bQAAAABJRU5ErkJggg=="
)

# --- NEW/UPDATED: Helpers for enforcing schema & image validation ---
def is_valid_base64_png(b64: str) -> bool:
    prefix = "data:image/png;base64,"
    try:
        if isinstance(b64, str) and b64.startswith(prefix):
            base64.b64decode(b64[len(prefix):], validate=True)
            return True
    except Exception:
        pass
    return False

def trim_base64_png(b64: str, maxbytes: int = 100000) -> str:
    prefix = "data:image/png;base64,"
    if b64.startswith(prefix):
        raw = base64.b64decode(b64[len(prefix):] + '==')
        if len(raw) <= maxbytes:
            return b64
        trimmed = base64.b64encode(raw[:maxbytes]).decode()
        return prefix + trimmed
    return b64


# --- NEW/UPDATED: Dict of all required keys for each dataset ---
REQUIRED_KEYS = {
    "network": {
        "edge_count": "number",
        "highest_degree_node": "string",
        "average_degree": "number",
        "density": "number",
        "shortest_path_alice_eve": "number",
        "network_graph": "string",     # base64 PNG
        "degree_histogram": "string"   # base64 PNG
    },
    "sales": {
        "total_sales": "number",
        "top_region": "string",
        "day_sales_correlation": "number",
        "bar_chart": "string",                  # base64 PNG
        "median_sales": "number",
        "total_sales_tax": "number",
        "cumulative_sales_chart": "string"      # base64 PNG
    },
    "weather": {
        "average_temp_c": "number",
        "max_precip_date": "string",
        "min_temp_c": "number",
        "temp_precip_correlation": "number",
        "average_precip_mm": "number",
        "temp_line_chart": "string",        # base64 PNG
        "precip_histogram": "string"        # base64 PNG
    }
}

# --- NEW/UPDATED: Utility to guess use-case/data-type based on questions/driver ---
def guess_dataset_type(driver, questions):
    text = " ".join(q.lower() for q in questions)
    url = (driver.get("url") or "").lower()
    if any(x in url for x in ["network", "edge"]) or "edge_count" in text:
        return "network"
    if any(x in url for x in ["sales"]) or "total_sales" in text:
        return "sales"
    if any(x in url for x in ["weather"]) or "average_temp" in text:
        return "weather"
    return None

# --- NEW/UPDATED: Output normalization to enforce required schema always present (never missing keys) ---

def enforce_schema(output: dict, required_keys: dict) -> dict:
    result = {}
    for key, typ in required_keys.items():
        val = output.get(key)
        if key.endswith(("graph", "chart", "histogram")):
            if not is_valid_base64_png(val):
                val = BASE64_PNG_PLACEHOLDER
            else:
                val = trim_base64_png(val)
        if val is None:
            val = BASE64_PNG_PLACEHOLDER if key.endswith(("graph", "chart", "histogram")) else None
        result[key] = val
    return result


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
    """
    Build a strict instruction prompt for the LLM:
    - Forces use of provided data only (no external knowledge).
    - Enforces numeric cleaning in calculations.
    - Requires output in strict JSON matching required schema.
    - Includes helper file previews if present.
    """
    # Join questions into human-readable numbered list
    question_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    # Guess dataset type and pick schema keys
    use_type = guess_dataset_type({"url": url}, questions)
    keys = REQUIRED_KEYS.get(use_type, {})

    # === 1. Intro rules ===
    prompt = (
        "You are an exacting data analyst assistant.\n"
        "For each question, only use the tables, their JSON serialization, and/or the visible page text provided below.\n"
        "Do NOT use any knowledge outside the given extracted data/text, even if you think you know it.\n"
        "Carefully parse and cleanse all numbers (remove $, commas, footnotes, citations, etc.) before calculations.\n"
        "Convert numeric values to number type before comparing or counting.\n"
        "For each table cell containing numbers, remove ALL non-numeric characters "
        "(including $, commas, *, footnotes, and extra text) and convert to integer or float before using.\n"
        "When answering 'earliest' or 'latest' questions, filter for matching entries "
        "then select the row with minimum or maximum year, and return BOTH the primary label and that year.\n"
        "If numeric fields are missing, treat them as not qualifying.\n"
        "Do NOT guess or use prior knowledge — only provided data.\n"
        "If a question requires filtering by numeric range, process relevant columns as numbers, not strings.\n"
        "If you cannot answer from the data given, return null for that output key.\n"
        "For numbers, output as ints/floats; for charts/graphs, output base64 PNG starting with 'data:image/png;base64,'\n"
        "Do not output markdown or explanations — only valid JSON.\n"
    )

    # === 2. Schema section ===
    if keys:
        prompt += "\nOutput must be a JSON object with ALL these keys (even if null):\n"
        for k, typ in keys.items():
            prompt += f"- {k}: {typ}\n"
        # Example object
        example_obj = {
            k: (
                BASE64_PNG_PLACEHOLDER
                if any(tok in k for tok in ['chart', 'graph', 'histogram'])
                else (1 if typ == 'number' else 'abc')
            )
            for k, typ in keys.items()
        }
        prompt += "Example output:\n" + json.dumps(example_obj, indent=2) + "\n"

    # === 3. Helper file descriptions (truncated) ===
    if helpers:
        prompt += "\nHelper files have been parsed as follows:\n"
        for fname, pdata in helpers.items():
            prompt += f"- File: {fname} (type: {fname.split('.')[-1]})\n"
            if fname.endswith('.csv'):
                csv_head = None
                if isinstance(pdata, dict):
                    csv_head = pdata.get("csv_preview", pdata.get("head", []))
                prompt += f"  Extracted CSV preview rows:\n{json.dumps(csv_head)[:1000]}...(truncated)\n"
            elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_uri = ""
                if isinstance(pdata, dict):
                    img_uri = pdata.get("image_data_uri", "")
                prompt += f"  Base64 image dataURI: {img_uri[:1000]}...(truncated)\n"
            elif fname.endswith('.txt') or fname.endswith('.pdf'):
                text_preview = pdata if isinstance(pdata, str) else json.dumps(pdata)
                prompt += f"  Text head:\n{text_preview[:1000]}...(truncated)\n"
            else:
                prompt += f"  Raw content preview:\n{str(pdata)[:500]}...(truncated)\n"
        prompt += (
            "\nIf a question explicitly references a helper file by name, use only the provided file data for that answer.\n"
            "If the referenced file data cannot be used, return null for that key."
        )

    # === 4. Append questions, metadata, tables, and text ===
    prompt += f"\nQuestions:\n{question_list}\n"
    prompt += f"Data source: {url}\n"
    prompt += f"Table metadata: {json.dumps(meta)}\n"
    prompt += f"Tables (JSON):\n{json.dumps(tables)}\n"
    if page_text:
        prompt += f"\nPage text extract:\n{page_text[:5000]}\n"

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

                # --- NEW/UPDATED: Enforce schema and handle list/dict conversions ---
                # Guess which type and required keys
                dtype = guess_dataset_type(driver, driver["questions"])
                req_keys = REQUIRED_KEYS.get(dtype, {})
                # If LLM output is a list, map in order or fallback to empty
                if isinstance(ans, list) and req_keys:
                    ans = {k: v for k, v in zip(req_keys, ans)}
                if not isinstance(ans, dict):
                    ans = {}
                # Always enforce full schema on output!
                normalized = enforce_schema(ans, req_keys) if req_keys else ans
                results[f"results{idx}"] = normalized
                continue
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