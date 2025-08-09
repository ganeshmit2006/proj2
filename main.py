import os
import traceback
import json
import re
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from fastapi import FastAPI, Request
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile
import base64

import duckdb
from bs4 import BeautifulSoup
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("Warning: Set GEMINI_API_KEY environment variable for LLM access.")
genai.configure(api_key=GEMINI_API_KEY)

def clean_gemini_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
       
    try:
        return json.loads(text)
    except Exception:
        array_match = re.search(r'(\\[.*?\\])', text, flags=re.DOTALL)
        object_match = re.search(r'(\{.*?\})', text, flags=re.DOTALL)
        if array_match:
            return json.loads(array_match.group(1))
        if object_match:
            return json.loads(object_match.group(1))
    return None

def response_shape_matches(parsed, expected_len):
    return (isinstance(parsed, list) and len(parsed) == expected_len) or (isinstance(parsed, dict) and len(parsed) == expected_len)

def parse_analytic_question_with_llm(question, columns):
    prompt = f"""
You are a data analyst assistant.
Given the columns available: {columns}
Generate a valid DuckDB SQL query that answers the question exactly.
Enclose the parquet source as {{DATA_PATH}} in the FROM clause.
Answer all questions by referring ONLY to the provided tables. Return a valid JSON array of answers, in order, with no explanations or markdown.
Question: {question}

Return ONLY the query string, no explanations.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    query = response.text.strip().strip('"')
    return query

def execute_duckdb_query(s3_path, query):
    try:
        conn = duckdb.connect(database=':memory:')
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL parquet; LOAD parquet;")
        # Add S3 credentials here if needed
        # conn.execute("SET s3_access_key_id='...';")
        # conn.execute("SET s3_secret_access_key='...';")

        if "{DATA_PATH}" in query:
            query = query.replace("{DATA_PATH}", f"'{s3_path}'")

        df = conn.execute(query).df()
        return df, None
    except Exception as e:
        return None, str(e)

def handle_duckdb_analytics(s3_path, questions):
    columns = [
        "court", "year", "date_of_registration", "decision_date", "disposal_nature",
        "title", "description", "judge", "pdf_link", "cnr", "court_code", "raw_html", "bench"
    ]
    results = {}
    for q in questions:
        query = parse_analytic_question_with_llm(q, columns)
        df, err = execute_duckdb_query(s3_path, query)
        if err:
            results[q] = {"error": err, "query": query}
            continue
        # scalar result
        if df.shape[0] == 1 and df.shape[1] == 1:
            results[q] = df.iloc[0, 0]
        # empty result
        elif df.shape[0] == 0:
            results[q] = []
        # plot result if requested
        elif "plot" in q.lower() and df.shape[1] >= 2:
            buf = io.BytesIO()
            xcol, ycol = df.columns[:2]
            df.plot(kind='scatter', x=xcol, y=ycol, title=q)
            plt.xlabel(xcol)
            plt.ylabel(ycol)
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            img = base64.b64encode(buf.getvalue()).decode('utf8')
            results[q] = f"data:image/png;base64,{img}"
        else:
            results[q] = df.to_dict(orient='records')
    return results

def extract_page_text(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts, styles, nav, aside, headers, footers for final page text
    for tag in soup(['script', 'style', 'nav', 'aside', 'footer', 'header']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    # Optionally only grab largest visible block or main section here
    return text

async def handle_scrape_and_llm(url, questions):
    from playwright.async_api import async_playwright
    SCRAPED_FILE = "scraped_content.html"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        content = await page.content()
        with open(SCRAPED_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        await browser.close()

    with open(SCRAPED_FILE, encoding="utf-8") as f:
        html = f.read()

    def extract_table_data(html):
        soup = BeautifulSoup(html, "html.parser")
        tables = []
        for table in soup.select("table.wikitable"):
            rows = table.select("tr")
            content = []
            for row in rows:
                cols = []
                for cell in row.find_all(["th", "td"]):
                    for tag in cell.find_all(True):
                        tag.decompose()
                    cols.append(cell.get_text(strip=True))
                if cols:
                    content.append(cols)
            if content:
                tables.append(content)
        return tables

    def extract_table_metadata(tables_data):
        metas = []
        for i, tdata in enumerate(tables_data):
            metas.append({
                "table_index": i,
                "num_rows": len(tdata),
                "num_cols": max((len(r) for r in tdata), default=0),
                "headers": tdata if tdata else [],
            })
        return {"num_tables": len(tables_data), "tables": metas}

    tables_data = extract_table_data(html)
    meta = extract_table_metadata(tables_data)

    # Add: Fallback to visible page text if no tables are extracted
    page_text = None
    if not tables_data or all(len(tbl) == 0 for tbl in tables_data):
        page_text = extract_page_text(html)

    prompt = build_agent_prompt(url, tables_data, meta, questions, page_text)
    llm_raw = call_gemini(prompt)
    print("Prompt to Gemini:", prompt)
    print("Raw response from Gemini:", repr(llm_raw))
    parsed = clean_gemini_response(llm_raw)

    if parsed is None:
        raise ValueError(f"LLM did not return valid JSON: {llm_raw}")
    if not response_shape_matches(parsed, len(questions)):
        if isinstance(parsed, list):
            while len(parsed) < len(questions):
                parsed.append(None)
            parsed = parsed[:len(questions)]
        elif isinstance(parsed, dict):
            for q in questions:
                if q not in parsed:
                    parsed[q] = None
    return parsed

def build_agent_prompt(url, tables_data, meta, questions, page_text=None, helper_files=None):
    questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = (
        "You are an exacting data analyst assistant.\n"
        "For each question, only use the tables and their JSON serialization and/or the visible page text provided below.\n"
        "Do NOT use any knowledge outside the given extracted data/text, even if you think you know it.\n"
        "Carefully parse and cleanse all numbers (remove $ signs, commas, footnotes, or citations etc) before calculations. "
        "Convert numeric values to number type before comparing or counting.\n"
        "For each table cell containing numbers, remove ALL non-numeric characters (including $, commas, *, footnotes, and extra text) and convert the result to an integer or float before using.\n"
        "When answering 'earliest' or 'latest' questions, filter for matching entries (like box office > $1.5bn), then select the row with the minimum or maximum year, and return BOTH the movie title and year for that row.\n"
        "If a numeric column is ambiguous or missing, treat missing values as not qualifying.\n"
        "Do NOT use any prior or external knowledge, only use what is found in the extracted tables and/or text below.\n"
        "Parse and convert all gross values to plain numbers before filtering.\n"
        "If a question requires finding a title or answer based on numeric ranges, process the relevant columns as numbers, not strings.\n"
        "Return only a valid JSON array or object (no markdown, no explanations, no code block, just valid JSON).\n"
        "If you cannot answer a question from the data given, respond with null for that output.\n"
        "For count/number answers, output integers, not strings. If summarizing, keep formats numeric and strict. For plots, output a base64 PNG data URI as a string.\n"
        "Answer all questions using ONLY the provided tables and/or page text.\n"
        "Additionally, you may use any uploaded helper file listed below (CSV, PNG, TXT, PDF etc) for answering questions that reference them explicitly by name.\n"
        "Helper files have been parsed as follows:"
    )
    if helper_files:
        for fname, pdata in helper_files.items():
            prompt += f"\n- File: {fname} (type: {fname.split('.')[-1]})\n"
            if fname.endswith('.csv'):
                prompt += f"  Extracted CSV rows:\n{json.dumps(pdata.get('csv_head', []))}\n"
            elif fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg'):
                # Only include the first 1,000 chars of the base64 URI to reduce length
                img_uri = pdata.get('image_data_uri', '')
                prompt += f"  Base64 image dataURI: {img_uri[:1000]}...(truncated)\n"
            elif fname.endswith('.txt'):
                txt_preview = pdata if isinstance(pdata, str) else json.dumps(pdata)
                prompt += f"  Text head:\n{txt_preview[:1000]}...(truncated)\n"
            elif fname.endswith('.pdf'):
                pdf_text = pdata if isinstance(pdata, str) else json.dumps(pdata)
                prompt += f"  PDF extracted text:\n{pdf_text[:1000]}...(truncated)\n"
            else:
                prompt += f"  Raw content preview:\n{str(pdata)[:500]}...(truncated)\n"
        prompt += (
            "\nIf a question explicitly references a helper file by name, use only the data provided above for your answer."
            " If the referenced file data cannot be used to answer, return null for that question."
            " Do not use prior knowledge or guess."
        )
    prompt += (
        f"\nQuestions:\n{questions_str}\n"
        f"Data source: {url}\n"
        f"Table metadata: {json.dumps(meta)}\n"
        f"Tables:\n{json.dumps(tables_data)}\n"
    )
    if page_text:
        prompt += f"\nPage text:\n{page_text[:500000]}\n"
    return prompt

def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


async def extract_url_and_questions_from_text(text: str):
    """
    Try to extract URL and questions from raw text heuristically.
    If heuristic extraction fails, use LLM to parse text into JSON.
    """
    # Heuristic extraction: find first url-like pattern
    url_match = re.search(r'https?://[^\s]+', text)
    url = url_match.group(0) if url_match else None

    # Try to extract question lines starting with number or bullet
    question_lines = re.findall(r'^\s*(?:\d+\.|-|\*)\s*(.+)', text, flags=re.MULTILINE)
    questions = [q.strip() for q in question_lines if q.strip()]

    if url and questions:
        return {"url": url, "questions": questions}

    # As a fallback, call LLM to parse better (optional but recommended)
    try:
        prompt = f"""
You are a helpful extractor agent.
Given the following free text input, extract the URL and an array of user questions from it.
Respond ONLY as JSON with keys: "url" (string) and "questions" (list of strings).

Input text:
\"\"\"
{text}
\"\"\"
"""

        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        parsed = json.loads(resp.text)
        # Basic validation
        if "url" in parsed and "questions" in parsed and isinstance(parsed["questions"], list):
            return parsed
    except Exception as e:
        print("LLM extraction fallback failed:", e)

    # If nothing useful parsed, raise
    raise ValueError("Unable to extract 'url' and 'questions' from text input.")


from typing import List
from fastapi import File, UploadFile

@app.post("/api/")
async def analyze(request: Request):
    """
    Accepts 0, 1, or multiple uploaded files with arbitrary field names.
    Also supports direct JSON/text body input if no files are sent.
    Processes all files in one request, selects driver file for answering.
    """
    try:
        files_info = []
        parsed_files = {}      # Store results per file
        main_data = None       # Store first driver file found

        # ---- File collection: this block unchanged ----
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            for field_name, value in form.multi_items():
                if isinstance(value, StarletteUploadFile):
                    content = await value.read()
                    filename = value.filename.lower()
                    content_type = value.content_type
                    files_info.append((field_name, filename, content, content_type))
        else:
            # No files uploaded, just body
            raw_body = await request.body()
            files_info.append(("body", "body", raw_body, "text/plain"))

        # ---- Main change: parse all files individually ----
        for field_name, filename, content, content_type in files_info:
            parsed_data = None

            # JSON files
            if filename.endswith(".json") or content_type == "application/json":
                try:
                    parsed_data = json.loads(content.decode(errors="ignore"))
                except Exception:
                    pass

            # TXT files
            elif filename.endswith(".txt"):
                text = content.decode(errors="ignore").strip()
                try:
                    parsed_data = json.loads(text)
                except Exception:
                    parsed_data = await extract_url_and_questions_from_text(text)

            # CSV files
            elif filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                parsed_data = {"csv_head": df.head().to_dict(orient="records"),
                               "csv_rows": df.shape[0],
                               "csv_cols": df.shape[1]}

            # PDF files
            elif filename.endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                pdf_text = "\n".join([page.extract_text() for page in reader.pages])
                parsed_data = await extract_url_and_questions_from_text(pdf_text)

            # ----- IMAGE HANDLING (highlighted)—returns base64 data URI -----
            elif content_type.startswith("image/"):
                encoded_img = base64.b64encode(content).decode('utf-8')
                data_uri = f"data:{content_type};base64,{encoded_img}"
                max_base64_length = 100000 * 4 // 3  # max for 100KB binary as base64
                if len(encoded_img) > max_base64_length:
                    encoded_img = encoded_img[:max_base64_length]
                    data_uri = f"data:{content_type};base64,{encoded_img}"
                parsed_data = {"image_data_uri": data_uri}
            # -----------------------------------------------------------

            else:
                # Fallback for other types
                text = content.decode(errors="ignore").strip()
                parsed_data = await extract_url_and_questions_from_text(text)

            parsed_files[filename] = parsed_data

            # ---- Change: pick the FIRST driver file for answering ----
            if (parsed_data
                and isinstance(parsed_data, dict)
                and (("url" in parsed_data and "questions" in parsed_data)
                     or ("s3_path" in parsed_data and "questions" in parsed_data))
                and not main_data):
                main_data = parsed_data

        # If nobody matched, return all file results for debugging
        if not main_data:
            return JSONResponse(status_code=400,
                                content={
                                    "error": "Invalid request. Provide 's3_path' + 'questions' or 'url' + 'questions'.",
                                    "files_parsed": parsed_files,
                                })

        # ---- Driver logic unchanged ----
        if "s3_path" in main_data and "questions" in main_data:
            results = handle_duckdb_analytics(main_data["s3_path"], main_data["questions"])
            return JSONResponse(content=results)
        elif "url" in main_data and "questions" in main_data:
            results = await handle_scrape_and_llm(main_data["url"], main_data["questions"])
            # Attach helper file results for answer context
            context = {k:v for k,v in parsed_files.items() if v and k != 'body' and k != filename}
            return JSONResponse(content={"results": results})
            #return JSONResponse(content={"results": results, "files_context": context})


        return JSONResponse(status_code=400,
                            content={
                                "error": "Invalid request. Provide 's3_path' + 'questions' or 'url' + 'questions'.",
                                "files_parsed": parsed_files,
                            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=81)

