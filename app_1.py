import os
import io
import json
import time
import pickle
import random
import numpy as np
import streamlit as st

from dotenv import load_dotenv
import fitz  # PyMuPDF

# ------ Setup ------
st.set_page_config(page_title="PA 211 RAG — Notebook Modes", layout="wide")
load_dotenv()

# Prefer Streamlit Secrets on Cloud, fallback to env
OPENAI_API_KEY = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY")

# Writable data dir (Cloud-safe)
DATA_DIR = os.environ.get("STREAMLIT_DATA_DIR", "/mount/data")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_GUIDE_PATHS = [
    os.path.join(DATA_DIR, "PA211_expanded_dataset.json"),
    "PA211_expanded_dataset.json",
    "/mnt/data/PA211_expanded_dataset.json",
]

# ------ Simple utils ------
def ensure_api_key(service_name, key):
    if not key:
        raise ValueError(f"{service_name} API key is missing. Add it in the sidebar or Secrets.")
    return key

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def retry_wait(attempt, base=6, cap=90):
    return min(cap, base * (attempt + 1))

# ------ Embedding cache (per-model, per-source) ------
def cache_path(model_name, source_name):
    safe = model_name.lower().replace("/", "_")
    return os.path.join(DATA_DIR, f"{safe}__{source_name}.pkl")

def load_cache(model_name, source_name):
    p = cache_path(model_name, source_name)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_cache(model_name, source_name, embeddings, texts):
    p = cache_path(model_name, source_name)
    try:
        with open(p, "wb") as f:
            pickle.dump({"embeddings": embeddings, "texts": texts}, f)
    except Exception as e:
        st.error(f"Failed to save cache for {model_name} ({source_name}): {e}")

# ------ PDF helpers ------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)

def chunk_text(text: str, chunk_size=1000, overlap=200):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ------ OpenAI/Gemini: embeddings & generation ------
def embed_openai_one(text: str, model="text-embedding-3-small"):
    ensure_api_key("OpenAI", OPENAI_API_KEY)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(6):
        try:
            resp = client.embeddings.create(model=model, input=[text])
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            time.sleep(retry_wait(attempt))
    return np.zeros((1536,), dtype=np.float32)  # safe fallback size for t-e-3-small

def embed_openai_many(texts, model="text-embedding-3-small"):
    return np.vstack([embed_openai_one(t) for t in texts]) if texts else np.zeros((0, 1536), dtype=np.float32)

def parse_gemini_embed(resp):
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list):
            return emb
    try:
        return resp.embedding.values  # type: ignore
    except Exception:
        pass
    raise ValueError("Unexpected Gemini embedding shape")

def embed_gemini_one(text: str, model="models/embedding-001"):
    ensure_api_key("Gemini", GEMINI_API_KEY)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    for attempt in range(6):
        try:
            resp = genai.embed_content(model=model, content=text)
            return np.array(parse_gemini_embed(resp), dtype=np.float32)
        except Exception:
            time.sleep(retry_wait(attempt))
    return np.zeros((768,), dtype=np.float32)  # typical dim for gemini emb-001

def embed_gemini_many(texts, model="models/embedding-001"):
    embs = []
    for i, t in enumerate(texts):
        embs.append(embed_gemini_one(t, model=model))
        if i and i % 20 == 0:
            time.sleep(5)
    if not embs:
        return np.zeros((0, 768), dtype=np.float32)
    return np.vstack(embs)

def call_openai(prompt: str, model="gpt-4o-mini", temperature=0.0):
    ensure_api_key("OpenAI", OPENAI_API_KEY)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer using only the provided context. If unsure, say you don't know."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(retry_wait(attempt))
    return "ERROR: generation failed."

def call_gemini(prompt: str, model="gemini-2.0-flash", temperature=0.0):
    ensure_api_key("Gemini", GEMINI_API_KEY)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    mdl = genai.GenerativeModel(model)
    for attempt in range(6):
        try:
            resp = mdl.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=temperature))
            return (resp.text or "").strip()
        except Exception:
            time.sleep(retry_wait(attempt))
    return "ERROR: generation failed."

# ------ Retrieval on cached embeddings ------
def build_or_load_cache(source: str, api: str, texts_fn, texts_args=None):
    """
    source: 'dataset' or 'pdfs'
    api: 'OpenAI' or 'Gemini'
    texts_fn: function that returns (texts, source_name_for_cache)
    """
    texts, source_name = texts_fn(*(texts_args or []))
    model_name = "text-embedding-3-small" if api == "OpenAI" else "models/embedding-001"

    cached = load_cache(model_name, source_name)
    if cached and cached.get("texts") == texts:
        return cached["embeddings"], texts

    st.sidebar.info(f"Building embeddings for {api} on {source_name}…")
    if api == "OpenAI":
        embs = embed_openai_many(texts)
    else:
        embs = embed_gemini_many(texts)
    save_cache(model_name, source_name, embs, texts)
    st.sidebar.success(f"Saved cache for {api} / {source_name}")
    return embs, texts

def retrieve_similar(query: str, embeddings: np.ndarray, texts: list, api: str, k: int):
    q_emb = embed_openai_one(query) if api == "OpenAI" else embed_gemini_one(query)
    sims = (embeddings @ q_emb) / (np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-9) + 1e-9)
    order = np.argsort(sims)[::-1][:k]
    return order, sims[order]

# ------ Context builders from sources ------
def load_dataset_texts():
    # prefer an existing JSON path
    for p in DEFAULT_GUIDE_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = [f"{item.get('question','')}\n{item.get('ideal_answer','')}".strip() for item in data]
            return texts, f"dataset_{os.path.basename(p)}"
    # if not found, return empty
    return [], "dataset_empty"

def pdfs_to_texts(pdfs: list, chunk_size=1000, overlap=200):
    texts = []
    for f in pdfs:
        name = getattr(f, "name", "uploaded.pdf")
        raw = extract_text_from_pdf_bytes(f.read())
        chunks = chunk_text(raw, chunk_size, overlap)
        for i, ch in enumerate(chunks):
            if ch.strip():
                texts.append(f"[{name}#chunk_{i}]\n{ch}")
    return texts, f"pdfs_{len(pdfs)}files_{chunk_size}_{overlap}"

# ------ Notebook modes (pipelines) ------
def context_enriched_answer(api: str, query: str, top_texts: list):
    # Build a brief then answer
    joined = "\n\n".join(top_texts)
    brief_prompt = f"Create a concise brief (5-8 bullets) capturing key facts needed to answer.\n\nQuestion: {query}\n\nContext:\n{joined}\n\nBrief:"
    brief = call_openai(brief_prompt) if api == "OpenAI" else call_gemini(brief_prompt)
    final_context = f"Brief:\n{brief}\n\n---\n\n{joined}"
    final_prompt = f"Answer the question using only the context.\n\nContext:\n{final_context}\n\nQuestion: {query}\nAnswer:"
    return call_openai(final_prompt) if api == "OpenAI" else call_gemini(final_prompt)

def qt_rewrite(api: str, query: str):
    p = f"Rewrite this query to be clearer and specific. Keep it short.\nQuery: {query}"
    return call_openai(p) if api == "OpenAI" else call_gemini(p)

def qt_step_back(api: str, query: str):
    p = f"Produce a single step-back question (the higher-level question that helps answer the original).\nOriginal: {query}\nStep-back:"
    return call_openai(p) if api == "OpenAI" else call_gemini(p)

def qt_decompose(api: str, query: str, n=3):
    p = f"Break the query into {n} short sub-questions (one per line).\nQuery: {query}\nSub-questions:"
    raw = call_openai(p) if api == "OpenAI" else call_gemini(p)
    subs = [s.strip("- ").strip() for s in raw.splitlines() if s.strip()]
    return subs[:n] if subs else [query]

def query_transform_answer(api: str, query: str, embeddings: np.ndarray, texts: list, k: int, mode: str):
    if mode == "rewrite":
        t = qt_rewrite(api, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, api, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        return (call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)), {"transformed": t}
    elif mode == "step_back":
        t = qt_step_back(api, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, api, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        return (call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)), {"step_back": t}
    else:
        subs = qt_decompose(api, query, n=min(3, k))
        # distribute retrieval across subs
        per = max(1, k // max(1, len(subs)))
        all_idxs = []
        for s in subs:
            idxs, _ = retrieve_similar(s, embeddings, texts, api, per)
            all_idxs.extend(list(idxs))
        # dedupe
        seen = set(); uniq = []
        for i in all_idxs:
            if i not in seen:
                uniq.append(i); seen.add(i)
        ctx = "\n\n".join([texts[i] for i in uniq[:k]])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)
        return ans, {"sub_questions": subs}

def classify_query(api: str, query: str):
    p = f"""Classify the query into exactly one category:
- Factual
- Analytical
- Opinion
- Contextual

Query: {query}
Return ONLY the category name.
"""
    out = call_openai(p) if api == "OpenAI" else call_gemini(p)
    for c in ["Factual", "Analytical", "Opinion", "Contextual"]:
        if c.lower() in (out or "").lower():
            return c
    return "Factual"

def adaptive_answer(api: str, query: str, embeddings: np.ndarray, texts: list, k: int, user_ctx: str):
    qtype = classify_query(api, query)
    if qtype == "Factual":
        t = qt_rewrite(api, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, api, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
    elif qtype == "Analytical":
        subs = qt_decompose(api, query, n=min(3, k))
        per = max(1, k // max(1, len(subs)))
        all_idxs = []
        for s in subs:
            idxs, _ = retrieve_similar(s, embeddings, texts, api, per)
            all_idxs.extend(list(idxs))
        seen = set(); uniq = []
        for i in all_idxs:
            if i not in seen:
                uniq.append(i); seen.add(i)
        ctx = "\n\n".join([texts[i] for i in uniq[:k]])
    elif qtype == "Opinion":
        angles = qt_decompose(api, f"Suggest 3 distinct viewpoints on: {query}", n=3)
        all_idxs = []
        for a in angles:
            idxs, _ = retrieve_similar(f"{query} {a}", embeddings, texts, api, 1)
            all_idxs.extend(list(idxs))
        ctx = "\n\n".join([texts[i] for i in all_idxs[:k]])
    else:  # Contextual
        reform = call_openai(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0) if api=="OpenAI" else call_gemini(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0)
        idxs, _ = retrieve_similar(reform, embeddings, texts, api, k)
        ctx = "\n\n".join([texts[i] for i in idxs])

    prompt = f"Answer using only the context.\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    ans = call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)
    return ans, qtype, ctx

# ------ UI: Sidebar ------
st.sidebar.header("API Keys")
openai_key_in = st.sidebar.text_input("OpenAI API Key", type="password", value=OPENAI_API_KEY or "")
gemini_key_in = st.sidebar.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY or "")
if openai_key_in:
    os.environ["OPENAI_API_KEY"] = openai_key_in
    OPENAI_API_KEY = openai_key_in
if gemini_key_in:
    os.environ["GEMINI_API_KEY"] = gemini_key_in
    GEMINI_API_KEY = gemini_key_in

st.sidebar.header("Source")
source_choice = st.sidebar.radio("Choose source", ["Dataset (Prompt Guide)", "PDFs"], index=0)

# Prompt-Guide loader
st.sidebar.subheader("Prompt Guide")
pg_file = st.sidebar.file_uploader("Upload PA211 JSON (optional)", type=["json"])
pg_load_btn = st.sidebar.button("Load Prompt Guide")

if "prompt_guide" not in st.session_state:
    # try default paths
    loaded = None
    for p in DEFAULT_GUIDE_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                break
            except Exception:
                pass
    st.session_state.prompt_guide = loaded if isinstance(loaded, list) else []

if pg_load_btn:
    try:
        if pg_file is not None:
            st.session_state.prompt_guide = json.load(pg_file)
        else:
            # re-attempt default paths
            data, _ = load_dataset_texts()
            st.session_state.prompt_guide = data
    except Exception as e:
        st.sidebar.error(f"Failed to load JSON: {e}")

# PDF area
st.sidebar.subheader("PDF Settings")
chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 1000, 100)
overlap = st.sidebar.number_input("Overlap", 0, 1000, 200, 50)
pdf_uploads = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# ------ UI: Main ------
st.title("PA 211 RAG — Notebook Modes")
st.caption("Modes: Context-Enriched, Query-Transform (rewrite/step-back/decompose), Adaptive")

# Fill from prompt guide
col_q, col_fill = st.columns([3, 1])
with col_q:
    query = st.text_area("Your question", key="query", height=100)
with col_fill:
    if st.session_state.prompt_guide:
        labels = [f"#{i+1}: {str(item.get('question',''))[:60]}" for i, item in enumerate(st.session_state.prompt_guide)]
        pick = st.selectbox("Prompt guide pick", list(range(len(labels))), format_func=lambda i: labels[i])
        if st.button("Use selected"):
            item = st.session_state.prompt_guide[pick]
            st.session_state.query = item.get("question", "")
            st.session_state.reference = item.get("ideal_answer", "")
            st.success("Filled query/reference from prompt guide.")
    else:
        st.info("Load a prompt guide (JSON) in the sidebar to enable quick fill.")

reference = st.text_area("(Optional) Reference answer — used for scoring", key="reference", height=120)

col_opts1, col_opts2 = st.columns(2)
with col_opts1:
    mode = st.selectbox("Notebook Mode", ["Context-Enriched", "Query-Transform", "Adaptive"])
    qt_mode = st.selectbox("If Query-Transform, choose:", ["rewrite", "step_back", "decompose"])
with col_opts2:
    top_k = st.slider("Top-K retrieved", 1, 10, 4)
    user_ctx = st.text_input("User context (for Adaptive/Contextual)")

run_oai = st.button("Run with OpenAI")
run_gem = st.button("Run with Gemini")
run_both = st.button("Compare Side-by-Side")

# ------ Prepare source texts & caches ------
def _dataset_texts_fn():
    # Return (texts, source_name)
    texts, src_name = load_dataset_texts()
    # If prompt guide is in session, prefer that
    if st.session_state.prompt_guide:
        texts = [f"{i.get('question','')}\n{i.get('ideal_answer','')}".strip() for i in st.session_state.prompt_guide]
        src_name = f"dataset_session_{len(texts)}"
    return texts, src_name

def _pdf_texts_fn():
    if not pdf_uploads:
        return [], "pdfs_empty"
    return pdfs_to_texts(pdf_uploads, chunk_size, overlap)

def build_source_embeddings(api):
    if source_choice == "Dataset (Prompt Guide)":
        return build_or_load_cache("dataset", api, _dataset_texts_fn)
    else:
        return build_or_load_cache("pdfs", api, _pdf_texts_fn)

# ------ Runner ------
def run_api(api: str):
    embeddings, texts = build_source_embeddings(api)
    if len(texts) == 0:
        st.error("No texts found. Load a dataset or upload PDFs.")
        return None

    if mode == "Context-Enriched":
        idxs, _ = retrieve_similar(query, embeddings, texts, api, top_k)
        picked = [texts[i] for i in idxs]
        ans = context_enriched_answer(api, query, picked)
        ctx = "\n\n".join(picked)
    elif mode == "Query-Transform":
        ans, info = query_transform_answer(api, query, embeddings, texts, top_k, qt_mode)
        # retrieve the actual context used:
        # quick re-retrieve based on info to score fairly
        if "transformed" in info:
            idxs, _ = retrieve_similar(info["transformed"], embeddings, texts, api, top_k)
            ctx = "\n\n".join([texts[i] for i in idxs])
        elif "step_back" in info:
            idxs, _ = retrieve_similar(info["step_back"], embeddings, texts, api, top_k)
            ctx = "\n\n".join([texts[i] for i in idxs])
        else:
            # for decompose, just rebuild from subs
            subs = info.get("sub_questions", [])
            per = max(1, top_k // max(1, len(subs)))
            all_idxs = []
            for s in subs:
                idxs, _ = retrieve_similar(s, embeddings, texts, api, per)
                all_idxs.extend(list(idxs))
            uniq = []
            seen = set()
            for i in all_idxs:
                if i not in seen:
                    uniq.append(i); seen.add(i)
            ctx = "\n\n".join([texts[i] for i in uniq[:top_k]])
    else:
        ans, qtype, ctx = adaptive_answer(api, query, embeddings, texts, top_k, user_ctx)

    # scoring: vs reference else vs context (OpenAI embedding space for consistency)
    try:
        ans_emb = embed_openai_one(ans)
        tgt = (reference or "").strip() or ctx
        tgt_emb = embed_openai_one(tgt)
        score = cosine_sim(ans_emb, tgt_emb)
    except Exception:
        score = 0.0

    return {"answer": ans, "score": score, "context": ctx}

# ------ Display ------
if run_oai or run_both:
    with st.container():
        st.subheader("OpenAI")
        res = run_api("OpenAI")
        if res:
            st.write(res["answer"])
            st.markdown(f"**Similarity score:** `{res['score']:.3f}`")
            with st.expander("Context used"):
                st.code(res["context"][:4000], language="markdown")

if run_gem or run_both:
    with st.container():
        st.subheader("Gemini")
        res = run_api("Gemini")
        if res:
            st.write(res["answer"])
            st.markdown(f"**Similarity score:** `{res['score']:.3f}`")
            with st.expander("Context used"):
                st.code(res["context"][:4000], language="markdown")

st.markdown("---")
st.caption("Built in the style of your prompt-guide app, but wired to your notebook modes (Context-Enriched, Query-Transform, Adaptive) with dataset/PDF sources and per-API embedding caches.")
