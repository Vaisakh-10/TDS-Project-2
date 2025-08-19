from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
import aiohttp, traceback, json, re, contextlib, asyncio
import pytesseract, base64, io, os
from PIL import Image
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
from starlette.datastructures import UploadFile as StarletteUploadFile
import json

# ===== PRIMARY AI CONFIG =====
pytesseract.pytesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
AI_PIPE_URL = os.getenv("AI_PIPE_URL", "https://api.groq.com/openai/v1/chat/completions")
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {AI_PIPE_TOKEN}", "Content-Type": "application/json"}
AI_DEBUG_LOG = "ai_response_debug.log"

# ===== FALLBACK AI CONFIG =====
FALLBACK_AI_PIPE_URL = os.getenv("FALLBACK_AI_PIPE_URL", "https://aipipe.org/openai/v1/chat/completions")
FALLBACK_AI_PIPE_TOKEN = os.getenv("FALLBACK_AI_PIPE_TOKEN", "")
FALLBACK_HEADERS = {
    "Authorization": f"Bearer {FALLBACK_AI_PIPE_TOKEN}",
    "Content-Type": "application/json"
}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ===== Security =====
BANNED_PATTERNS = [
    r"import\s+os", r"import\s+sys", r"import\s+subprocess",
    r"os\.", r"sys\.", r"subprocess\.", r"open\(", r"exec\(", r"eval\(",
    r"__import__", r"shutil\.", r"socket\."
]
BANNED_REGEX = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]

# ===== Helpers =====

async def alternate_process_request(question: str, main_table: pd.DataFrame = None, attachments=None):
    
    if attachments is None:
        attachments = []
    all_tables = []
    column_mapping = {}
    schema = ""

    local_vars = {
        "shared_vars": {},
        "main_table": main_table,
        "all_tables": all_tables,
        "pd": pd,
        "np": np,
        "json": json,
        "flatten_multiindex_columns": flatten_multiindex_columns,
        "sanitize_columns_for_ai": sanitize_columns_for_ai,
        "fig_to_base64": fig_to_base64,
        "nx": nx,
        "plt": plt,
        "BytesIO": BytesIO,
        "base64": base64,
    }

    output_format = detect_output_format_and_keys(question)
    try:
        steps = await request_ai_steps(question, schema=f"Columns: {list(main_table.columns) if main_table is not None else None}, shape: {main_table.shape if main_table is not None else None}")
        repairs = []
        prev_steps = []
        for idx, step in enumerate(steps, start=1):
            await run_step_with_self_heal(step["code"], idx, local_vars, repairs, prev_steps)
            # Debug log after each step
            print(f"[DEBUG] After step {idx}, text_answer=", local_vars.get("text_answer"))

        final_answer = local_vars.get("text_answer")
        if final_answer:
            try:
                parsed = json.loads(final_answer)
                print("[DEBUG] Final JSON answer:", parsed)
                return parsed
            except Exception:
                print("[DEBUG] Final non-JSON answer:", final_answer)
                return {"answer": final_answer}
        else:
            print("[DEBUG] Alternate process also failed to produce output")
            return {"error": "Alternate process also failed to produce output"}

    except Exception as e:
        # fallback direct answer mode
        raw_answer = await direct_answer_ai(question, main_table=main_table, output_schema=output_format)
        processed = postprocess_output(raw_answer, output_format)
        try:
            parsed = json.loads(processed)
            print("[DEBUG] Fallback JSON answer:", parsed)
            return parsed
        except Exception:
            print("[DEBUG] Fallback non-JSON answer:", processed)
            return {"answer": processed}


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    fig.clf()
    return b64

def ensure_chart_keys(result_dict, chart_key_funcs):
    for key, chart_func in chart_key_funcs.items():
        val = result_dict.get(key)
        try:
            if not (val and isinstance(val, str)):
                raise ValueError
            img_bytes = base64.b64decode(val)
            if img_bytes[:8] != b'\x89PNG\r\n\x1a\n' or len(img_bytes) >= 100*1024:
                raise ValueError
        except Exception:
            fig = chart_func()
            encoded_img = fig_to_base64(fig)
            tries = 0
            while len(base64.b64decode(encoded_img)) >= 100*1024 and tries < 5:
                fig.set_size_inches(fig.get_size_inches()*0.8)
                encoded_img = fig_to_base64(fig)
                tries += 1
            result_dict[key] = encoded_img
    return result_dict

def flatten_multiindex_columns(df):
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(" | ".join(str(x).strip() for x in c))
        else:
            new_cols.append(str(c).strip())
    df.columns = new_cols
    return df

def sanitize_columns_for_ai(df):
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', col).strip('_') for col in df.columns]
    return df

def log_ai_response(reason, resp_text):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(AI_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n--- [{ts}] {reason} ---\n{resp_text}\n")

def safe_extract_ai_content(data, context="AI"):
    if not isinstance(data, dict):
        raise RuntimeError(f"{context}: bad API format")
    choice = data.get("choices", [{}])[0].get("message", {})
    if "content" not in choice:
        raise RuntimeError(f"{context}: missing content")
    return choice["content"]

def get_last_debug_lines(n=25):
    if not os.path.exists(AI_DEBUG_LOG):
        return ""
    with open(AI_DEBUG_LOG, encoding="utf-8") as f:
        return "".join(f.readlines()[-n:])

def sanitize_code(raw):
    code = re.sub(r"(?i)^here is .*?:", "", raw).strip()
    code = re.sub(r"(?im)^```", "", code)
    code = re.sub(r"(?m)^```", "", code)
    code = re.sub(r"```", "", code)
    if code.startswith(("'''", '"""')) and code.endswith(("'''", '"""')):
        code = code.strip("'\"`").strip()
    return code.strip()

def detect_output_format_and_keys(question: str):
    q_lower = question.lower()
    m = re.search(r'return a json object with keys:\s*((?:\n|.)+?)(?:\n[a-z0-9].*|$)', q_lower, re.I)
    if m:
        keys_block = m.group(1)
        # Accept lines like - `key`: ... as well as inline usage
        keys = re.findall(r'`([^`]+)`', keys_block)
        return {"type": "json", "fields": keys}

    if "only a number" in q_lower or "numeric only" in q_lower:
        return {"type": "number"}
    if "csv" in q_lower:
        return {"type": "csv"}
    if "table" in q_lower:
        return {"type": "table"}
    return {"type": "text"}

def postprocess_output(raw_text_answer: str, output_format: dict):
    from json import JSONDecodeError
    if output_format["type"] == "json":
        try:
            parsed = json.loads(raw_text_answer)
            if isinstance(parsed, dict) and "fields" in output_format:
                parsed_fixed = {key: parsed.get(key, None) for key in output_format["fields"]}
                return json.dumps(parsed_fixed)
        except JSONDecodeError:
            return json.dumps({"result": raw_text_answer})
    elif output_format["type"] == "number":
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', raw_text_answer)
        return m.group(0) if m else "NaN"
    elif output_format["type"] == "csv":
        if "," in raw_text_answer or "\n" in raw_text_answer:
            return raw_text_answer
        else:
            return "value\n" + raw_text_answer
    elif output_format["type"] == "table":
        return raw_text_answer
    else:
        return raw_text_answer

# ===== Chart Generators (examples) =====
def generate_bar_chart(df):
    fig, ax = plt.subplots()
    if "region" in df and "sales" in df:
        df.groupby("region")["sales"].sum().plot(kind="bar", color="blue", ax=ax)
        ax.set_xlabel("Region")
        ax.set_ylabel("Total Sales")
    return fig

def generate_cumulative_sales_chart(df):
    fig, ax = plt.subplots()
    if "date" in df and "sales" in df:
        df_sorted = df.sort_values("date")
        df_sorted["cum_sales"] = df_sorted["sales"].cumsum()
        ax.plot(df_sorted["date"], df_sorted["cum_sales"], color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Sales")
    return fig

def generate_network_graph(G):
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, ax=ax)
    return fig

def generate_degree_histogram(G):
    fig, ax = plt.subplots()
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, color="green")
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Frequency")
    return fig

# ===== Table scraper =====
def fetch_tables_and_clean(url):
    html = requests.get(url, timeout=15).text
    tables = pd.read_html(html)
    cleaned = []
    for df in tables:
        df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
        cleaned.append(df)
    chosen_df = max(cleaned, key=lambda x: x.shape * x.shape[1])
    return cleaned, chosen_df

# ===== AI generation =====
async def request_ai_steps(question, attempt=1, max_attempts=2, debug_chain=None, schema=""):
    if debug_chain is None:
        debug_chain = []
    schema_note = f"\n\nSchema:\n{schema}" if schema else ""
    schema_req = """The JSON MUST be:
{"steps":[{"description":"...","code":"Python code"}]}"""
    base_prompt = """
RETURN ONLY VALID JSON. No prose. No markdown. No triple quotes.
The dataset is already loaded as a pandas DataFrame named 'main_table'. Do NOT use pd.read_csv or try to read data from disk.
You must follow the output format exactly as requested by the user:
- If user requests JSON with keys, output ONLY that JSON.
- If CSV format requested, output exactly CSV with headers.
- If a single numeric value requested, output ONLY the number.
- If a table is requested, output as markdown table only.
Do NOT add any explanations or additional text.
If a chart/base64 image is requested, ALWAYS use fig_to_base64(matplotlib_figure) to encode the plot, so it's a base64 PNG string under 100 kB.
Rules:
1. Persistent Python environment — variables/imports stay between steps.
2. main_table has sanitized string columns.
3. Print main_table.columns before references.
4. Convert to numeric with pd.to_numeric(errors='coerce') before calculations.
5. Skip gracefully if column missing.
6. No unsafe imports.
7. The final step MUST set local_vars["text_answer"] to a JSON string containing the final answer as per user's request.
8. All values in the dictionary assigned to local_vars["text_answer"] MUST be Python base types (int, float, str) and NOT NumPy types—convert using .item() or float()/int()/str().
"""
    q_lower = question.lower()
    if "json" in q_lower:
        base_prompt += "\nIMPORTANT: Your output MUST be valid JSON ONLY — no prose or extra text!"
    elif "csv" in q_lower:
        base_prompt += "\nIMPORTANT: Your output MUST be CSV with a header row only — no extra text!"
    elif "only a number" in q_lower or "numeric only" in q_lower:
        base_prompt += "\nIMPORTANT: Your output MUST be JUST a number with no other text!"
    elif "table" in q_lower:
        base_prompt += "\nIMPORTANT: Your output MUST be a markdown table only — no prose or explanations!"
    base_prompt += f"\n{schema_req}\n\nUser request:\n{question}\n{schema_note}"
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a careful Python data analyst."},
            {"role": "user", "content": base_prompt}
        ]
    }

    print("[DEBUG] LLM primary (request_ai_steps) prompt being sent:\n", payload["messages"][-1]["content"])
    print("[DEBUG] LLM primary payload schema:\n", f"main_table columns: {schema}")

    async with aiohttp.ClientSession() as s:
        async with s.post(AI_PIPE_URL, headers=HEADERS, json=payload) as r:
            data = await r.json()
            raw = safe_extract_ai_content(data, "generate").strip()
            print("[DEBUG] LLM primary RAW AI RESPONSE:\n", raw)
            log_ai_response("RAW AI OUTPUT", raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                log_ai_response("JSON PARSE ERROR", raw)
                print("[DEBUG] JSON PARSE ERROR in LLM response:", raw)
                debug_chain.append("json_error")
                if attempt < max_attempts:
                    return await request_ai_steps(question, attempt + 1, max_attempts, debug_chain, schema)
                return [{"description": "Fallback", "code": sanitize_code(raw)}]
            steps = parsed.get("steps", [])
            for st in steps:
                st["code"] = sanitize_code(st.get("code", ""))
            return steps

async def generate_fixed_code(orig_code, err_trace, step_number, local_vars, prev_steps):
    prompt = f"""
RETURN ONLY VALID PYTHON CODE. No prose. No markdown/backticks. No triple quotes.
Step {step_number} failed.
Error:
{err_trace}
Original code:
{orig_code}
Rules:
- Persistent env: variables/imports exist from earlier.
- main_table columns are sanitized strings.
- Always print(main_table.columns) first.
- Convert numeric data safely, handle NaN, skip missing cols.
- Only final step should assign to local_vars["text_answer"] as JSON string of answers.
"""
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You fix broken Python code for data analysis."},
            {"role": "user", "content": prompt}
        ]
    }

    async with aiohttp.ClientSession() as s:
        async with s.post(AI_PIPE_URL, headers=HEADERS, json=payload) as r:
            data = await r.json()
            fixed = safe_extract_ai_content(data, "repair").strip()
            return sanitize_code(fixed)

async def run_step_with_self_heal(step_code, step_number, local_vars, repairs, prev_steps):
    try:
        exec(step_code, {}, local_vars)
        prev_steps.append({"status": "success", "code": step_code})
    except Exception:
        err = traceback.format_exc()
        fixed = await generate_fixed_code(step_code, err, step_number, local_vars, prev_steps)
        repairs.append({
            "step_number": step_number,
            "original_code": step_code,
            "error": err,
            "fixed_code": fixed
        })
        try:
            exec(fixed, {}, local_vars)
            prev_steps.append({"status": "success", "code": fixed})
        except Exception as e2:
            repairs[-1]["repair_failed"] = str(e2)
            prev_steps.append({"status": "fail", "code": fixed})
            raise

# ===== Fallback Direct Answer AI =====
async def direct_answer_ai(question, main_table=None, output_schema=None):
    data_info = ""
    if main_table is not None:
        sample_csv = main_table.head(20).to_csv(index=False)
        print("[DEBUG] Sample CSV sent to LLM:\n", sample_csv)
        data_info = f"\nDataset (CSV sample):\n{sample_csv}"
    format_info = ""
    if output_schema is not None:
        if output_schema.get("type") == "json" and "fields" in output_schema:
            format_info = f"\nIMPORTANT: The answer MUST be returned as a JSON object with keys: {output_schema['fields']}. Return ONLY the JSON object, not an explanation or markdown!"
        elif output_schema.get("type") == "csv":
            format_info = "\nIMPORTANT: Return ONLY the answer as CSV with header, no additional explanation, markdown, or prose."
        elif output_schema.get("type") == "number":
            format_info = "\nIMPORTANT: Return ONLY a number, with no other text."
        elif output_schema.get("type") == "table":
            format_info = "\nIMPORTANT: Return ONLY as markdown table."
    fallback_prompt = (
        "You are a data analyst. A user has submitted the following open-ended data analysis task."
        "\nProvide ONLY the final answer, not the steps, in exactly the output format required."
        f"\n\nUser question/request:\n{question}"
        f"{data_info}"
        f"{format_info}"
    )
    fallback_payload = {
        "model": "gpt-4.1",  # Change if your fallback model is different
        "messages": [
            {"role": "system", "content": "You answer data analysis questions with direct results, using available data if any."},
            {"role": "user", "content": fallback_prompt}
        ]
    }
    
    print("[DEBUG] LLM fallback prompt being sent:\n", fallback_prompt)
    print("[DEBUG] LLM fallback payload schema:\n", f"main_table columns: {main_table.columns if main_table is not None else None}")

    async with aiohttp.ClientSession() as s:
        async with s.post(FALLBACK_AI_PIPE_URL, headers=FALLBACK_HEADERS, json=fallback_payload) as r:
            data = await r.json()
            raw = safe_extract_ai_content(data, "fallback").strip()
            print("[DEBUG] LLM fallback RAW AI RESPONSE:\n", raw)
            log_ai_response("FALLBACK RAW AI OUTPUT", raw)
            return raw

# ===== API =====
@app.post("/api/")
@app.post("/api/")
async def analyze(request: Request):
    form = await request.form()
    print(f"[DEBUG] Form keys: {list(form.keys())}")

    # Debug print all form keys and their value types/content
    for k, v in form.items():
        print(f"[DEBUG] Form item: key={k}, value type={type(v)}, value={repr(v)[:100]}")

    question = ""
    main_table = None
    attachments = []

    for key, value in form.items():
        print(f"[DEBUG] Processing form key={key}, value type={type(value)}")
        fname = getattr(value, 'filename', '').lower() if hasattr(value, 'filename') else ''
        lower_key = key.lower()
        print(f"[DEBUG] value.filename: {fname}")
        # Accept any reasonable filename/key for the question file
        if (
            (lower_key in ["questions.txt", "question.txt"])
            or (fname in ["questions.txt", "question.txt"])
            or lower_key.startswith("question") or fname.startswith("question")
        ):
            print(f"[DEBUG] Entered question file read block (key: {lower_key}, filename: {fname})")
            await value.seek(0)
            contents = await value.read()
            print(f"[DEBUG] Read {len(contents)} bytes from question file")
            question = contents.decode(errors="ignore").strip()
            print(f"[DEBUG] Decoded question length: {len(question)}")
        elif fname.endswith('.csv') or lower_key.endswith('.csv'):
            contents = await value.read()
            try:
                df = pd.read_csv(io.BytesIO(contents))
                print("[DEBUG] LOADED DATAFRAME HEAD:", df.head())
                df = sanitize_columns_for_ai(df)
                for c in df.columns:
                    if "date" in c.lower():
                        try:
                            df[c] = pd.to_datetime(df[c])
                        except Exception:
                            pass
                main_table = df
            except Exception:
                pass
        elif fname.endswith((".png", ".jpg", ".jpeg")):
            contents = await value.read()
            try:
                img_text = pytesseract.image_to_string(Image.open(io.BytesIO(contents))).strip()
                if question:
                    question += "\n" + img_text
                else:
                    question = img_text
            except Exception:
                pass
        else:
            # other attachments
            if hasattr(value, "filename"):
                contents = await value.read()
                attachments.append((fname, contents))

    print(f"[DEBUG] Question value: {question!r}, main_table is None: {main_table is None}")

    if not question and main_table is None:
        return {"error": "No input"}

    return await process_request(question, main_table, attachments)

async def process_request(question: str, main_table: pd.DataFrame = None, attachments=None):
    if attachments is None:
        attachments = []
    all_tables = []
    column_mapping = {}
    schema = ""

    with open("all_questions.txt", "a", encoding="utf-8") as qf:
        qf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {question}\n")

    print(f"[LOGGED QUESTION] {question}")

    m = re.search(r'(https?://\S+)', question)
    all_tables = []
    column_mapping = {}
    schema = ""
    if main_table is None and m:
        all_tables, main_table = fetch_tables_and_clean(m.group(1))
        main_table = flatten_multiindex_columns(main_table)
        original_columns = list(main_table.columns)
        main_table = sanitize_columns_for_ai(main_table)
        sanitized_columns = list(main_table.columns)
        column_mapping = dict(zip(original_columns, sanitized_columns))
        log_ai_response("COLUMN MAPPING", json.dumps(column_mapping, ensure_ascii=False, indent=2))
        log_ai_response("SANITIZED COLUMNS", str(main_table.columns))
        schema = f"Columns: {list(main_table.columns)}, shape: {main_table.shape}"
    elif main_table is not None:
        schema = f"Columns: {list(main_table.columns)}, shape: {main_table.shape}"

    forced_steps = [{
        "description": "Ensure numeric conversion for all columns",
        "code": """
        print('Sanitized columns:', main_table.columns)
        for col in main_table.columns:
            try:
                main_table[col] = pd.to_numeric(main_table[col], errors='coerce')
            except Exception:
                pass
        print('Columns after conversion:', main_table.columns)
        """,
    }]

    debug_chain, repairs, prev_steps = [], [], []
    ai_steps = await request_ai_steps(question, debug_chain=debug_chain, schema=schema)
    steps = forced_steps + ai_steps

    local_vars = {
        "shared_vars": {},
        "main_table": main_table,
        "all_tables": all_tables,
        "pd": pd,
        "np": np,
        "json": json,
        "flatten_multiindex_columns": flatten_multiindex_columns,
        "sanitize_columns_for_ai": sanitize_columns_for_ai,
        "fig_to_base64": fig_to_base64
    }

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for i, step in enumerate(steps, 1):
            code = step.get("code", "")
            if any(rgx.search(code) for rgx in BANNED_REGEX):
                continue
            try:
                print("[DEBUG] Executing code step:\n", code)
                print(f"[DEBUG] Executing code step #{i}: {code[:120]}...")
                await run_step_with_self_heal(code, i, local_vars, repairs, prev_steps)
                print("[DEBUG] step executed OK")
            except Exception as e:
                print("[DEBUG] Exception in exec step:", e)
                print(f"[DEBUG] Exception running step #{i}: {e}")
                import traceback
                print(traceback.format_exc())
                continue
    
    print("[DEBUG] local_vars after step execution:", local_vars)
    # Unified extraction logic: handle possible "text_answer" nesting
    def extract_text_answer(local_vars):
        ta = local_vars.get("text_answer")
        # If not found at top-level, try nested
        if ta is None and "local_vars" in local_vars and isinstance(local_vars["local_vars"], dict):
            ta = local_vars["local_vars"].get("text_answer")
        return ta

    text_answer = extract_text_answer(local_vars)
    print("[DEBUG] text_answer (unified lookup):", text_answer)

    final_answer = str(text_answer if text_answer is not None else "").strip()
    print("[DEBUG] final_answer (pre-postprocess):", repr(final_answer))
    print("[DEBUG] type(final_answer):", type(final_answer), final_answer)
    try:
        if isinstance(text_answer, dict):
            result_dict = text_answer
        else:
            result_dict = json.loads(final_answer)
    except Exception:
        result_dict = None

    output_schema = detect_output_format_and_keys(question)
    print(f"[DEBUG] Output schema detected from question: {output_schema}")

    try:
        result_dict = json.loads(final_answer)
    except Exception:
        result_dict = None

    chart_key_funcs = {}
    if result_dict:
        if "bar_chart" in result_dict:
            chart_key_funcs["bar_chart"] = lambda: generate_bar_chart(main_table)
        if "cumulative_sales_chart" in result_dict:
            chart_key_funcs["cumulative_sales_chart"] = lambda: generate_cumulative_sales_chart(main_table)
        if ("network_graph" in result_dict or "degree_histogram" in result_dict) and main_table is not None:
            if {"source", "target"}.issubset(set(main_table.columns)):
                G = nx.from_pandas_edgelist(main_table, source="source", target="target")
                if "network_graph" in result_dict:
                    chart_key_funcs["network_graph"] = lambda: generate_network_graph(G)
                if "degree_histogram" in result_dict:
                    chart_key_funcs["degree_histogram"] = lambda: generate_degree_histogram(G)
        result_dict = ensure_chart_keys(result_dict, chart_key_funcs)
        final_answer_processed = postprocess_output(json.dumps(result_dict), output_schema)
    else:
        final_answer_processed = postprocess_output(final_answer, output_schema)

    use_fallback_model = False
    if not final_answer_processed or "Traceback" in final_answer_processed or "Error" in final_answer_processed:
        use_fallback_model = True

    if use_fallback_model:
        fallback_answer = await direct_answer_ai(question, main_table, output_schema)
        final_answer_processed = postprocess_output(fallback_answer, output_schema)

    if not final_answer_processed or "Traceback" in final_answer_processed or "Error" in final_answer_processed:
        error_msg = "Unable to produce a valid answer for your request."
        if output_schema["type"] == "json":
            final_answer_processed = json.dumps({"error": error_msg})
        elif output_schema["type"] == "number":
            final_answer_processed = "NaN"
        elif output_schema["type"] == "csv":
            final_answer_processed = "error\n" + error_msg
        elif output_schema["type"] == "table":
            final_answer_processed = f"| Error |\n|---|\n| {error_msg} |"
        else:
            final_answer_processed = error_msg

        with open("error_logs.txt", "a", encoding="utf-8") as ef:
            ef.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Request:\n{question}\n")
            ef.write(f"--- Raw text_answer ---\n{final_answer}\n")
            ef.write(f"--- Stdout Log ---\n{stdout.getvalue()}\n")
            ef.write(f"--- Repairs ---\n{json.dumps(repairs, ensure_ascii=False, indent=2)}\n")
            ef.write("-" * 50 + "\n")

    with open("logs.txt", "a", encoding="utf-8") as lf:
        lf.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Question: {question}\n")
        lf.write(f"--- LOG ---\n{stdout.getvalue()}\n")
        lf.write(f"--- REPAIRS ---\n{json.dumps(repairs, ensure_ascii=False, indent=2)}\n")
        lf.write(f"--- COLUMN MAPPING ---\n{json.dumps(column_mapping, ensure_ascii=False, indent=2)}\n")
        lf.write(f"{'-'*50}\n")

    output_schema = detect_output_format_and_keys(question)
    print(f"[DEBUG] Output schema detected from question: {output_schema}")

    # previous final_answer_processed computation...

    # --- existing return handling ---
    if output_schema["type"] == "json":
        try:
            response = json.loads(final_answer_processed)
        except Exception:
            response = final_answer_processed
    else:
        response = {"answer": final_answer_processed}

    # ====== NEW LOGIC: trigger alternate process if bad output ======
    bad_output = False
    if isinstance(response, dict):
        if response.get("error") == "No output generated":
            bad_output = True
        if "result" in response and str(response["result"]).strip() == "":
            bad_output = True
    elif isinstance(response, str):
        # in case JSON parsing failed, check raw string
        if response.strip() in ('{"error": "No output generated"}',
                                '{"result": ""}',
                                '{"answer": ""}'):
            bad_output = True

    if bad_output:
        print("[DEBUG] Bad output detected, running alternate_process_request")
        return await alternate_process_request(question, main_table, attachments)

    return response
