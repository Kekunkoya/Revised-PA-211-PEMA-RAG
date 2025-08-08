import os
import io
import re
import json
import time
import random
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF

# =========================
# App setup
# =========================
st.set_page_config(page_title="PA 211 RAG — Notebook Modes (OpenAI vs Gemini)", page_icon="🧪", layout="wide")
load_dotenv()

# Secrets / env only (no widgets)
OPENAI_API_KEY = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in secrets or environment.")
if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY not found in secrets or environment.")

# Writable data dir
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

# =========================
# Session state init (before widgets)
# =========================
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "reference" not in st.session_state:
    st.session_state["reference"] = ""
if "prompt_guide" not in st.session_state:
    st.session_state["prompt_guide"] = []


# =========================
# Utilities
# =========================
def ensure_api_key(service: str, key: Optional[str]):
    if not key:
        raise RuntimeError(f"{service} API key missing. Add it to Streamlit secrets or environment.")
    return key

def retry_wait(attempt: int, base: int = 6, cap: int = 90) -> int:
    return min(cap, base * (attempt + 1))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# Cache helpers (per-model, per-source)
def cache_path(model_name: str, source_name: str) -> str:
    safe = model_name.lower().replace("/", "_")
    return os.path.join(DATA_DIR, f"{safe}__{source_name}.pkl")

def load_cache(model_name: str, source_name: str):
    p = cache_path(model_name, source_name)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_cache(model_name: str, source_name: str, embeddings: np.ndarray, texts: List[str]):
    p = cache_path(model_name, source_name)
    try:
        with open(p, "wb") as f:
            pickle.dump({"embeddings": embeddings, "texts": texts}, f)
    except Exception as e:
        st.error(f"Failed to save cache for {model_name}/{source_name}: {e}")


# =========================
# PDF & chunking
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i+chunk_size])
    return chunks


# =========================
# Embeddings
# =========================
def embed_openai_one(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    ensure_api_key("OpenAI", OPENAI_API_KEY)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(6):
        try:
            resp = client.embeddings.create(model=model, input=[text])
            vec = resp.data[0].embedding
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            time.sleep(retry_wait(attempt))
    # Fallback vector (dimension for text-embedding-3-small)
    return np.zeros((1536,), dtype=np.float32)

def embed_openai_many(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    embs = [embed_openai_one(t, model=model) for t in texts]
    return np.vstack(embs)

def parse_gemini_embed(resp) -> List[float]:
    # Handle dict or object forms
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list):
            return emb
    try:
        return resp.embedding.values  # type: ignore[attr-defined]
    except Exception:
        pass
    raise ValueError("Unexpected Gemini embedding shape")

def embed_gemini_one(text: str, model: str = "models/embedding-001") -> np.ndarray:
    ensure_api_key("Gemini", GEMINI_API_KEY)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    for attempt in range(6):
        try:
            resp = genai.embed_content(model=model, content=text)
            vec = parse_gemini_embed(resp)
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            time.sleep(retry_wait(attempt))
    # Fallback (typical dimension)
    return np.zeros((768,), dtype=np.float32)

def embed_gemini_many(texts: List[str], model: str = "models/embedding-001") -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    embs = [embed_gemini_one(t, model=model) for t in texts]
    return np.vstack(embs)


# =========================
# LLM Generation
# =========================
def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    ensure_api_key("OpenAI", OPENAI_API_KEY)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer only with the provided context. If unsure, say you don't know."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(retry_wait(attempt))
    return "ERROR: generation failed."

def call_gemini(prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.0) -> str:
    ensure_api_key("Gemini", GEMINI_API_KEY)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    mdl = genai.GenerativeModel(model)
    for attempt in range(6):
        try:
            resp = mdl.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=temperature))
            return (resp.text or "").strip()
        except Exception as e:
            time.sleep(retry_wait(attempt))
    return "ERROR: generation failed."


# =========================
# Retrieval + pipelines
# =========================
def retrieve_similar(query: str, embeddings: np.ndarray, texts: List[str], api: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_emb = embed_openai_one(query) if api == "OpenAI" else embed_gemini_one(query)
    norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-9) + 1e-9
    sims = (embeddings @ q_emb) / norms
    order = np.argsort(sims)[::-1][:k]
    return order, sims[order]

def context_enriched_answer(api: str, query: str, top_texts: List[str]) -> str:
    joined = "\n\n".join(top_texts)
    brief_prompt = f"Create a concise brief (5-8 bullets) of key facts to answer.\n\nQuestion: {query}\n\nContext:\n{joined}\n\nBrief:"
    brief = call_openai(brief_prompt) if api == "OpenAI" else call_gemini(brief_prompt)
    final_ctx = f"Brief:\n{brief}\n\n---\n\n{joined}"
    prompt = f"Answer using only this context.\n\nContext:\n{final_ctx}\n\nQuestion: {query}\nAnswer:"
    return call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)

def qt_rewrite(api: str, query: str) -> str:
    p = f"Rewrite this query to be clearer and specific. Keep it short.\nQuery: {query}"
    return call_openai(p) if api == "OpenAI" else call_gemini(p)

def qt_step_back(api: str, query: str) -> str:
    p = f"Produce a single step-back question (the higher-level question that helps answer the original).\nOriginal: {query}\nStep-back:"
    return call_openai(p) if api == "OpenAI" else call_gemini(p)

def qt_decompose(api: str, query: str, n: int = 3) -> List[str]:
    p = f"Break the query into {n} short sub-questions (one per line).\nQuery: {query}\nSub-questions:"
    raw = call_openai(p) if api == "OpenAI" else call_gemini(p)
    subs = [s.strip("- ").strip() for s in raw.splitlines() if s.strip()]
    return subs[:n] if subs else [query]

def query_transform_answer(api: str, query: str, embeddings: np.ndarray, texts: List[str], k: int, mode: str):
    if mode == "rewrite":
        t = qt_rewrite(api, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, api, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)
        return ans, {"transformed": t}, ctx
    elif mode == "step_back":
        t = qt_step_back(api, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, api, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)
        return ans, {"step_back": t}, ctx
    else:
        subs = qt_decompose(api, query, n=min(3, k))
        per = max(1, k // max(1, len(subs)))
        all_idxs = []
        for s in subs:
            idxs, _ = retrieve_similar(s, embeddings, texts, api, per)
            all_idxs.extend(list(idxs))
        seen=set(); uniq=[]
        for i in all_idxs:
            if i not in seen:
                uniq.append(i); seen.add(i)
        ctx = "\n\n".join([texts[i] for i in uniq[:k]])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if api == "OpenAI" else call_gemini(prompt)
        return ans, {"sub_questions": subs}, ctx

def classify_query(api: str, query: str) -> str:
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

def adaptive_answer(api: str, query: str, embeddings: np.ndarray, texts: List[str], k: int, user_ctx: str):
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
        seen=set(); uniq=[]
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


# =========================
# Source building
# =========================
def load_dataset_texts() -> Tuple[List[str], str]:
    # Default paths
    for p in DEFAULT_GUIDE_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                texts = [f"{i.get('question','')}\n{i.get('ideal_answer','')}".strip() for i in data]
                return texts, f"dataset_{os.path.basename(p)}"
            except Exception:
                pass
    # From session prompt_guide if present
    if st.session_state.get("prompt_guide"):
        texts = [f"{i.get('question','')}\n{i.get('ideal_answer','')}".strip() for i in st.session_state["prompt_guide"]]
        return texts, f"dataset_session_{len(texts)}"

    return [], "dataset_empty"

def pdfs_to_texts(files: List[io.BytesIO], chunk_size=1000, overlap=200) -> Tuple[List[str], str]:
    texts = []
    for f in files:
        name = getattr(f, "name", "uploaded.pdf")
        raw = extract_text_from_pdf_bytes(f.read())
        chunks = chunk_text(raw, chunk_size, overlap)
        for i, ch in enumerate(chunks):
            if ch.strip():
                texts.append(f"[{name}#chunk_{i}]\n{ch}")
    return texts, f"pdfs_{len(files)}files_{chunk_size}_{overlap}"

def build_or_load_cache(source: str, api: str, texts_fn, texts_args=None) -> Tuple[np.ndarray, List[str]]:
    texts, source_name = texts_fn(*(texts_args or []))
    model_name = "text-embedding-3-small" if api == "OpenAI" else "models/embedding-001"

    cached = load_cache(model_name, source_name)
    if cached and cached.get("texts") == texts:
        return cached["embeddings"], texts

    st.sidebar.info(f"Building embeddings for {api} / {source_name}…")
    if api == "OpenAI":
        embs = embed_openai_many(texts)
    else:
        embs = embed_gemini_many(texts)
    save_cache(model_name, source_name, embs, texts)
    st.sidebar.success(f"Saved cache for {api} / {source_name}")
    return embs, texts


# =========================
# Prompt Guide callbacks
# =========================
def use_prompt_from_index(idx: int):
    items = st.session_state.get("prompt_guide") or []
    if not items or idx is None or idx < 0 or idx >= len(items):
        return
    item = items[idx]
    st.session_state["query"] = item.get("question", "")
    st.session_state["reference"] = item.get("ideal_answer", "")
    st.rerun()

def use_prompt_random():
    items = st.session_state.get("prompt_guide") or []
    if not items:
        return
    item = random.choice(items)
    st.session_state["query"] = item.get("question", "")
    st.session_state["reference"] = item.get("ideal_answer", "")
    st.rerun()


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Source & Prompt Guide")

    source_choice = st.radio("Choose source", ["Dataset (Prompt Guide)", "PDFs"], index=0)

    # Prompt guide loader
    pg_uploader = st.file_uploader("Upload prompt guide JSON", type=["json"])
    col_pg = st.columns(2)
    with col_pg[0]:
        load_pg = st.button("Load prompt guide")
    with col_pg[1]:
        clear_pg = st.button("Clear")

    if load_pg:
        try:
            loaded = None
            if pg_uploader is not None:
                loaded = json.load(pg_uploader)
            else:
                # try defaults
                for p in DEFAULT_GUIDE_PATHS:
                    if os.path.exists(p):
                        with open(p, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                        break
            if isinstance(loaded, list):
                st.session_state["prompt_guide"] = loaded
                st.success(f"Loaded {len(loaded)} items.")
            else:
                st.warning("JSON must be a list of {question, ideal_answer}.")
        except Exception as e:
            st.error(f"Failed to load prompt guide: {e}")
    if clear_pg:
        st.session_state["prompt_guide"] = []
        st.info("Prompt guide cleared.")

    if st.session_state.get("prompt_guide"):
        labels = [f"#{i+1}: {str(item.get('question',''))[:60]}" for i, item in enumerate(st.session_state["prompt_guide"])]
        sel_idx = st.selectbox("Pick a sample", list(range(len(labels))), format_func=lambda i: labels[i])
        col_sel = st.columns(2)
        with col_sel[0]:
            st.button("Use selected", on_click=use_prompt_from_index, args=(sel_idx,))
        with col_sel[1]:
            st.button("Use random", on_click=use_prompt_random)

    # PDF settings
    if source_choice == "PDFs":
        chunk_size = st.number_input("Chunk size", 200, 4000, 1000, 100)
        overlap = st.number_input("Overlap", 0, 1000, 200, 50)
        pdf_uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    else:
        chunk_size = 1000
        overlap = 200
        pdf_uploads = []

    # API pause
    pause_s = st.slider("Pause between API calls (rate-limit helper)", 0, 20, 6)


# =========================
# Main UI
# =========================
st.title("🧪 PA 211 RAG — Notebook Modes (OpenAI vs Gemini)")
st.caption("Modes: Context-Enriched, Query-Transform (rewrite / step-back / decompose), Adaptive")

query = st.text_area("Your question", key="query", height=100)
reference = st.text_area("(Optional) Reference answer — used for scoring", key="reference", height=120)

col_opts1, col_opts2 = st.columns(2)
with col_opts1:
    mode = st.selectbox("Notebook Mode", ["Context-Enriched", "Query-Transform", "Adaptive"])
    qt_mode = st.selectbox("If Query-Transform, choose:", ["rewrite", "step_back", "decompose"])
with col_opts2:
    top_k = st.slider("Top-K retrieved", 1, 10, 4)
    user_ctx = st.text_input("User context (for Adaptive/Contextual)")

col_run = st.columns(3)
with col_run[0]:
    run_oai = st.button("Run with OpenAI")
with col_run[1]:
    run_gem = st.button("Run with Gemini")
with col_run[2]:
    run_both = st.button("Compare side-by-side")

# Source text functions
def _dataset_texts_fn():
    texts, src = load_dataset_texts()
    return texts, src

def _pdf_texts_fn():
    if not pdf_uploads:
        return [], "pdfs_empty"
    return pdfs_to_texts(pdf_uploads, chunk_size, overlap)

def build_source_embeddings(api: str):
    if source_choice == "Dataset (Prompt Guide)":
        return build_or_load_cache("dataset", api, _dataset_texts_fn)
    else:
        return build_or_load_cache("pdfs", api, _pdf_texts_fn)

def score_answer(answer: str, fallback_context: str) -> float:
    # Score in OpenAI embedding space vs reference if provided else vs context
    if not OPENAI_API_KEY:
        return 0.0
    try:
        ans_emb = embed_openai_one(answer)
        tgt = (reference or "").strip() or fallback_context
        tgt_emb = embed_openai_one(tgt)
        return cosine_sim(ans_emb, tgt_emb)
    except Exception:
        return 0.0

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
        ans, info, ctx = query_transform_answer(api, query, embeddings, texts, top_k, qt_mode)
    else:
        ans, qtype, ctx = adaptive_answer(api, query, embeddings, texts, top_k, user_ctx)

    score = score_answer(ans or "", ctx or "")
    return {"answer": ans, "score": score, "context": ctx}

# Display
if run_oai or run_both:
    with st.container():
        st.subheader("OpenAI")
        res = run_api("OpenAI")
        if res:
            st.write(res["answer"])
            st.markdown(f"**Similarity score:** `{res['score']:.3f}`")
            with st.expander("Context used"):
                st.code(res["context"][:4000], language="markdown")

if run_both:
    time.sleep(pause_s)

if run_gem or run_both:
    with st.container():
        st.subheader("Gemini")
        res = run_api("Gemini")
        if res:
            st.write(res["answer"])
            st.markdown(f"**Similarity score:** `{res['score']:.3f}`")
            with st.expander("Context used"):
                st.code(res["context"][:4000], language="markdown")

# Optional cross-answer similarity (OpenAI space)
# Only show when both buttons pressed in same run; we can store last results in session if needed.
st.markdown("---")
st.caption("No API key widgets — keys must be provided via Streamlit Secrets or environment. Caches live in a writable data dir.")
