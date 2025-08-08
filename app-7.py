import os
import io
import re
import time
import json
import random
from typing import List, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
import numpy as np
import fitz  # PyMuPDF

# OpenAI SDK (v1+)
from openai import OpenAI
import openai

# Gemini SDK
import google.generativeai as genai
import google.api_core.exceptions


# =========================
# Setup
# =========================
st.set_page_config(page_title="RAG: OpenAI vs Gemini (Context-Enriched, Query-Transform, Adaptive)", page_icon="🧪", layout="wide")
load_dotenv()

# Prefer Streamlit Secrets on Cloud; fallback to env locally
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("Missing OPENAI_API_KEY (set in Streamlit Secrets or .env)")
if not GEMINI_API_KEY:
    st.warning("Missing GEMINI_API_KEY (set in Streamlit Secrets or .env)")

# Initialize clients if keys present
oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Writable data dir (works on Streamlit Cloud & local)
DATA_DIR = os.environ.get("STREAMLIT_DATA_DIR", "/mount/data")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    # fallback to local relative dir
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)

STORE_OPENAI = os.path.join(DATA_DIR, "vector_store_openai.pkl")
STORE_GEMINI = os.path.join(DATA_DIR, "vector_store_gemini.pkl")

# Try to pre-load prompt guide if present
PROMPT_GUIDE_DEFAULTS = [
    os.path.join(DATA_DIR, "PA211_expanded_dataset.json"),
    "PA211_expanded_dataset.json",
    "/mnt/data/PA211_expanded_dataset.json",
]
def try_load_json(paths: list) -> Optional[list]:
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            continue
    return None


# =========================
# Core: Simple Vector Store
# =========================
class SimpleVectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[dict] = []

    def add_item(self, text: str, embedding: List[float], metadata: Optional[dict] = None):
        self.vectors.append(np.array(embedding, dtype=np.float32))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict]:
        if not self.vectors:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        nq = np.linalg.norm(q)
        sims = []
        for i, v in enumerate(self.vectors):
            nv = np.linalg.norm(v)
            score = 0.0 if nq == 0 or nv == 0 else float(np.dot(q, v) / (nq * nv))
            sims.append((i, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        out = []
        for i in range(min(k, len(sims))):
            idx, score = sims[i]
            out.append({"text": self.texts[idx], "metadata": self.metadata[idx], "similarity": score})
        return out


# =========================
# Helpers: PDF, chunking
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)

def extract_text_from_pdf_path(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
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
# Embeddings: OpenAI vs Gemini
# =========================
def embed_openai_one(text: str, model: str = "text-embedding-3-small") -> List[float]:
    # Basic retry/backoff
    for attempt in range(6):
        try:
            resp = oai_client.embeddings.create(model=model, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            # catch broadly to avoid SDK version mismatch on error class names
            wait_s = min(60, 6 * (attempt + 1))
            st.info(f"[OpenAI embed] Error or rate limit; waiting {wait_s}s… ({e})")
            time.sleep(wait_s)
    return [0.0]

def embed_openai_many(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    embs = []
    for i, t in enumerate(texts):
        if not t.strip():
            embs.append([0.0])
            continue
        if i and i % 20 == 0:
            time.sleep(5)
        embs.append(embed_openai_one(t, model=model))
    return embs

def parse_gemini_embed(resp) -> List[float]:
    # Some versions return {'embedding': {'values': [...]}}; others use attributes
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
    raise ValueError("Unexpected Gemini embedding response shape")

def embed_gemini_one(text: str, model: str = "models/embedding-001") -> List[float]:
    for attempt in range(6):
        try:
            resp = genai.embed_content(model=model, content=text)
            return parse_gemini_embed(resp)
        except google.api_core.exceptions.ResourceExhausted as e:
            m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(e))
            wait_s = int(m.group(1)) if m else min(90, 8 * (attempt + 1))
            st.info(f"[Gemini embed] Rate limit; waiting {wait_s}s…")
            time.sleep(wait_s)
        except Exception as e:
            st.info(f"[Gemini embed] Error; retrying… {e}")
            time.sleep(min(90, 8 * (attempt + 1)))
    return [0.0]

def embed_gemini_many(texts: List[str], model: str = "models/embedding-001") -> List[List[float]]:
    embs = []
    for i, t in enumerate(texts):
        if not t.strip():
            embs.append([0.0])
            continue
        if i and i % 20 == 0:
            time.sleep(5)
        embs.append(embed_gemini_one(t, model=model))
    return embs


# =========================
# LLM Calls (generation) + helpers used by different pipelines
# =========================
def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    for attempt in range(6):
        try:
            resp = oai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use only the provided context. If unsure, say you don't know."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            wait_s = min(90, 8 * (attempt + 1))
            st.info(f"[OpenAI gen] Error or rate limit; waiting {wait_s}s… ({e})")
            time.sleep(wait_s)
    return "ERROR: generation failed."

def call_gemini(prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.2) -> str:
    mdl = genai.GenerativeModel(model)
    for attempt in range(6):
        try:
            resp = mdl.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=temperature))
            return (resp.text or "").strip()
        except google.api_core.exceptions.ResourceExhausted as e:
            m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(e))
            wait_s = int(m.group(1)) if m else min(90, 8 * (attempt + 1))
            st.info(f"[Gemini gen] Rate limit; waiting {wait_s}s…")
            time.sleep(wait_s)
        except Exception as e:
            st.info(f"[Gemini gen] Error; retrying… {e}")
            time.sleep(min(90, 8 * (attempt + 1)))
    return "ERROR: generation failed."


# =========================
# Pipelines
#   1) Context-Enriched RAG
#   2) Query Transform RAG (rewrite/step-back/decompose)
#   3) Adaptive RAG (classify → choose strategy)
# =========================

# ---- Query transforms ----
def qt_rewrite(query: str, provider: str) -> str:
    prompt = f"Rewrite this query to be clearer and more specific. Keep it short.\nQuery: {query}"
    if provider == "openai":
        return call_openai(prompt, temperature=0)
    else:
        return call_gemini(prompt, temperature=0)

def qt_step_back(query: str, provider: str) -> str:
    prompt = f"Produce a single 'step-back' question: the higher-level question whose answer helps answer the original.\nOriginal: {query}\nStep-back:"
    if provider == "openai":
        return call_openai(prompt, temperature=0)
    else:
        return call_gemini(prompt, temperature=0)

def qt_decompose(query: str, provider: str, n: int = 3) -> List[str]:
    prompt = f"Break the query into {n} short sub-questions (one per line).\nQuery: {query}\nSub-questions:"
    raw = call_openai(prompt, temperature=0) if provider == "openai" else call_gemini(prompt, temperature=0)
    subs = [s.strip("- ").strip() for s in raw.splitlines() if s.strip()]
    return subs[:n] if subs else [query]


# ---- Retrieval helpers ----
def retrieve(provider: str, query: str, store: SimpleVectorStore, k: int = 4):
    if provider == "openai":
        q_emb = embed_openai_one(query)
    else:
        q_emb = embed_gemini_one(query)
    return store.similarity_search(q_emb, k=k)

def generate(provider: str, query: str, passages: List[Dict]) -> str:
    context = "\n\n---\n\n".join([p["text"] for p in passages]) if passages else ""
    if provider == "openai":
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        return call_openai(prompt, temperature=0.2)
    else:
        prompt = f"You are a helpful assistant. Use only the provided context. If unsure, say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}"
        return call_gemini(prompt, temperature=0.2)

# 1) Context-Enriched RAG: summarize/condense retrieved passages first
def context_enriched(provider: str, query: str, store: SimpleVectorStore, k: int = 4) -> Tuple[str, List[Dict]]:
    hits = retrieve(provider, query, store, k=k)
    # Condense context
    brief = ""
    if hits:
        joined = "\n\n".join([h["text"] for h in hits])
        condenser_prompt = f"Create a concise brief (5-8 bullet points) capturing key facts needed to answer the question.\n\nQuestion: {query}\n\nContext:\n{joined}\n\nBrief:"
        brief = call_openai(condenser_prompt, temperature=0.2) if provider == "openai" else call_gemini(condenser_prompt, temperature=0.2)
        # use brief as additional leading context
        hits = [{"text": f"Brief:\n{brief}", "metadata": {"source":"brief"}, "similarity": 1.0}] + hits
    answer = generate(provider, query, hits)
    return answer, hits

# 2) Query-Transform RAG
def query_transform(provider: str, query: str, store: SimpleVectorStore, k: int = 4, mode: str = "rewrite") -> Tuple[str, List[Dict], Dict]:
    info = {}
    if mode == "rewrite":
        t = qt_rewrite(query, provider)
        info["transformed"] = t
        hits = retrieve(provider, t, store, k=k)
        answer = generate(provider, query, hits)
        return answer, hits, info
    elif mode == "step_back":
        t = qt_step_back(query, provider)
        info["step_back"] = t
        hits = retrieve(provider, t, store, k=k)
        answer = generate(provider, query, hits)
        return answer, hits, info
    else:  # decompose
        subs = qt_decompose(query, provider, n=3)
        info["sub_questions"] = subs
        all_hits = []
        for s in subs:
            all_hits += retrieve(provider, s, store, k=max(1, k//len(subs) or 1))
        # dedupe by text
        seen = set(); dedup = []
        for h in all_hits:
            if h["text"] not in seen:
                dedup.append(h); seen.add(h["text"])
        answer = generate(provider, query, dedup[:k])
        return answer, dedup[:k], info

# 3) Adaptive RAG
def classify_query(provider: str, query: str) -> str:
    prompt = f"""Classify the query into exactly one category:
- Factual
- Analytical
- Opinion
- Contextual

Query: {query}
Return ONLY the category name.
"""
    out = call_openai(prompt, temperature=0) if provider == "openai" else call_gemini(prompt, temperature=0)
    for c in ["Factual", "Analytical", "Opinion", "Contextual"]:
        if c.lower() in (out or "").lower():
            return c
    return "Factual"

def adaptive(provider: str, query: str, store: SimpleVectorStore, k: int = 4, user_ctx: Optional[str] = None) -> Tuple[str, List[Dict], str]:
    qtype = classify_query(provider, query)
    if qtype == "Factual":
        t = qt_rewrite(query, provider)  # light enhancement
        hits = retrieve(provider, t, store, k=k)
    elif qtype == "Analytical":
        subs = qt_decompose(query, provider, n=3)
        hits = []
        for s in subs:
            hits += retrieve(provider, s, store, k=max(1, k//len(subs) or 1))
        # dedupe
        seen=set(); dedup=[]
        for h in hits:
            if h["text"] not in seen:
                dedup.append(h); seen.add(h["text"])
        hits = dedup[:k]
    elif qtype == "Opinion":
        angles = qt_decompose(f"Suggest 3 distinct viewpoints on: {query}", provider, n=3)
        hits = []
        for a in angles:
            hits += retrieve(provider, f"{query} {a}", store, k=1)
    else:  # Contextual
        ctx_q = call_openai(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0) if provider=="openai" else call_gemini(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0)
        hits = retrieve(provider, ctx_q, store, k=k)
    answer = generate(provider, query, hits)
    return answer, hits, qtype


# =========================
# Save / Load Index
# =========================
def save_store(store: SimpleVectorStore, path: str):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(store, f)

def load_store(path: str) -> Optional[SimpleVectorStore]:
    import pickle
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, SimpleVectorStore) else None


# =========================
# Scoring (cosine similarity via OpenAI embeddings for consistency)
# =========================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def score_answer(answer: str, reference: Optional[str], fallback_context: Optional[str]) -> float:
    if not oai_client:
        return 0.0
    try:
        ans_emb = np.array(embed_openai_one(answer), dtype=np.float32)
        tgt = (reference or "").strip() or (fallback_context or "")
        tgt_emb = np.array(embed_openai_one(tgt), dtype=np.float32)
        return cosine_sim(ans_emb, tgt_emb)
    except Exception:
        return 0.0


# =========================
# UI
# =========================
st.title("🧪 RAG Comparison — OpenAI vs Gemini")
st.caption("Pipelines: Context-Enriched, Query-Transform (rewrite/step-back/decompose), Adaptive")

# --- Prompt Guide (dataset) ---
if "prompt_guide" not in st.session_state:
    st.session_state.prompt_guide = try_load_json(PROMPT_GUIDE_DEFAULTS) or []

with st.sidebar:
    st.header("Prompt guide (dataset)")
    st.caption("Load a JSON list with entries like: {'question': '...', 'ideal_answer': '...'}")

    pg_file = st.file_uploader("Upload prompt guide JSON", type=["json"])
    pg_path = st.text_input("Or JSON path", value=PROMPT_GUIDE_DEFAULTS[0])

    colpg1, colpg2 = st.columns(2)
    with colpg1:
        load_pg = st.button("Load prompt guide")
    with colpg2:
        clear_pg = st.button("Clear")

    if load_pg:
        loaded = None
        if pg_file is not None:
            try:
                loaded = json.load(pg_file)
            except Exception as e:
                st.warning(f"Could not parse uploaded JSON: {e}")
        elif pg_path:
            try:
                with open(pg_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except Exception as e:
                st.warning(f"Could not open {pg_path}: {e}")
        if isinstance(loaded, list):
            st.session_state.prompt_guide = loaded
            st.success(f"Loaded {len(loaded)} entries.")
        else:
            st.warning("JSON must be a list of objects with 'question' and (optionally) 'ideal_answer'.")
    if clear_pg:
        st.session_state.prompt_guide = []
        st.info("Prompt guide cleared.")

    if st.session_state.prompt_guide:
        questions = [str(item.get("question", f"Q{i+1}"))[:140] for i, item in enumerate(st.session_state.prompt_guide)]
        sel_idx = st.selectbox("Pick a sample question", list(range(len(questions))), format_func=lambda i: questions[i])
        colf1, colf2 = st.columns(2)
        with colf1:
            use_sel = st.button("Use selected")
        with colf2:
            use_rand = st.button("Use random")
        if use_sel and 0 <= sel_idx < len(st.session_state.prompt_guide):
            item = st.session_state.prompt_guide[sel_idx]
            st.session_state["query"] = item.get("question", "")
            st.session_state["reference"] = item.get("ideal_answer", "")
            st.success("Filled the fields with the selected item.")
        if use_rand:
            item = random.choice(st.session_state.prompt_guide)
            st.session_state["query"] = item.get("question", "")
            st.session_state["reference"] = item.get("ideal_answer", "")
            st.success("Filled the fields with a random item.")

    st.header("Indexing")
    chunk_size = st.number_input("Chunk size", 200, 4000, 1000, 100)
    overlap = st.number_input("Overlap", 0, 1000, 200, 50)
    folder_path = st.text_input("Folder of PDFs (optional)", value=os.path.join(DATA_DIR, "pdfs"))
    uploads = st.file_uploader("Or upload PDFs", type=["pdf"], accept_multiple_files=True)
    build_oai = st.button("Build/Replace OpenAI Index")
    build_gem = st.button("Build/Replace Gemini Index")
    st.caption("Build both to compare side-by-side.")

    st.header("Load/Save")
    colA, colB = st.columns(2)
    with colA:
        save_oai = st.button("Save OpenAI Index")
    with colB:
        save_gem = st.button("Save Gemini Index")
    colC, colD = st.columns(2)
    with colC:
        load_oai = st.button("Load OpenAI Index")
    with colD:
        load_gem = st.button("Load Gemini Index")

# Session state stores
if "store_openai" not in st.session_state:
    st.session_state.store_openai = None
if "store_gemini" not in st.session_state:
    st.session_state.store_gemini = None

# Build indexes
def build_index_from_uploads(files: List[io.BytesIO], chunk_size=1000, overlap=200, provider="openai") -> SimpleVectorStore:
    store = SimpleVectorStore()
    for f in files:
        name = getattr(f, "name", "uploaded.pdf")
        text = extract_text_from_pdf_bytes(f.read())
        chunks = chunk_text(text, chunk_size, overlap)
        embs = embed_openai_many(chunks) if provider=="openai" else embed_gemini_many(chunks)
        for i, (ch, emb) in enumerate(zip(chunks, embs)):
            if ch.strip():
                store.add_item(ch, emb, {"source": name, "chunk_id": i})
    return store

def build_index_from_folder(folder: str, chunk_size=1000, overlap=200, provider="openai") -> SimpleVectorStore:
    store = SimpleVectorStore()
    if not os.path.isdir(folder):
        return store
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        text = extract_text_from_pdf_path(path)
        chunks = chunk_text(text, chunk_size, overlap)
        embs = embed_openai_many(chunks) if provider=="openai" else embed_gemini_many(chunks)
        for i, (ch, emb) in enumerate(zip(chunks, embs)):
            if ch.strip():
                store.add_item(ch, emb, {"source": fname, "chunk_id": i})
    return store

if build_oai:
    s = SimpleVectorStore()
    if uploads:
        s = build_index_from_uploads(uploads, chunk_size, overlap, provider="openai")
    elif folder_path and os.path.isdir(folder_path):
        s = build_index_from_folder(folder_path, chunk_size, overlap, provider="openai")
    st.session_state.store_openai = s
    st.success("OpenAI index built.")

if build_gem:
    s = SimpleVectorStore()
    if uploads:
        s = build_index_from_uploads(uploads, chunk_size, overlap, provider="gemini")
    elif folder_path and os.path.isdir(folder_path):
        s = build_index_from_folder(folder_path, chunk_size, overlap, provider="gemini")
    st.session_state.store_gemini = s
    st.success("Gemini index built.")

# Save/Load
if save_oai and st.session_state.store_openai:
    save_store(st.session_state.store_openai, STORE_OPENAI)
    st.success(f"OpenAI index saved → {STORE_OPENAI}")
if save_gem and st.session_state.store_gemini:
    save_store(st.session_state.store_gemini, STORE_GEMINI)
    st.success(f"Gemini index saved → {STORE_GEMINI}")

if load_oai:
    s = load_store(STORE_OPENAI)
    if s and s.vectors:
        st.session_state.store_openai = s
        st.success(f"Loaded OpenAI index ({len(s.vectors)} vectors).")
    else:
        st.warning("No saved OpenAI index found.")
if load_gem:
    s = load_store(STORE_GEMINI)
    if s and s.vectors:
        st.session_state.store_gemini = s
        st.success(f"Loaded Gemini index ({len(s.vectors)} vectors).")
    else:
        st.warning("No saved Gemini index found.")

st.divider()
st.header("Ask once → Compare side-by-side")

# Bind inputs to session_state so the prompt guide can fill them
query = st.text_input("Your question", key="query")
reference = st.text_area("(Optional) Reference answer for scoring", height=120, key="reference")
top_k = st.slider("Top-K passages", 1, 10, 4)
pipeline = st.selectbox("Pipeline", ["Context-Enriched RAG", "Query Transform RAG", "Adaptive RAG"], index=0)
qt_mode = st.selectbox("If Query Transform: mode", ["rewrite", "step_back", "decompose"], index=0, help="Only used when Pipeline=Query Transform RAG.")
user_ctx = st.text_input("Optional user context (used by Adaptive's 'Contextual' branch)")
pause_s = st.slider("Pause between API calls (sec) — increase if rate-limited", 0, 20, 6)
go = st.button("Run")

# --- Retrieval + Answer generation wrappers ---
def retrieve(provider: str, query: str, store: SimpleVectorStore, k: int = 4):
    if provider == "openai":
        q_emb = embed_openai_one(query)
    else:
        q_emb = embed_gemini_one(query)
    return store.similarity_search(q_emb, k=k)

def generate(provider: str, query: str, passages: List[Dict]) -> str:
    context = "\n\n---\n\n".join([p["text"] for p in passages]) if passages else ""
    if provider == "openai":
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        return call_openai(prompt, temperature=0.2)
    else:
        prompt = f"You are a helpful assistant. Use only the provided context. If unsure, say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}"
        return call_gemini(prompt, temperature=0.2)

# Pipelines (redeclared for scoping clarity)
def context_enriched(provider: str, query: str, store: SimpleVectorStore, k: int = 4) -> Tuple[str, List[Dict]]:
    hits = retrieve(provider, query, store, k=k)
    brief = ""
    if hits:
        joined = "\n\n".join([h["text"] for h in hits])
        condenser_prompt = f"Create a concise brief (5-8 bullet points) capturing key facts needed to answer the question.\n\nQuestion: {query}\n\nContext:\n{joined}\n\nBrief:"
        brief = call_openai(condenser_prompt, temperature=0.2) if provider == "openai" else call_gemini(condenser_prompt, temperature=0.2)
        hits = [{"text": f"Brief:\n{brief}", "metadata": {"source":"brief"}, "similarity": 1.0}] + hits
    answer = generate(provider, query, hits)
    return answer, hits

def query_transform(provider: str, query: str, store: SimpleVectorStore, k: int = 4, mode: str = "rewrite") -> Tuple[str, List[Dict], Dict]:
    info = {}
    if mode == "rewrite":
        t = qt_rewrite(query, provider)
        info["transformed"] = t
        hits = retrieve(provider, t, store, k=k)
        answer = generate(provider, query, hits)
        return answer, hits, info
    elif mode == "step_back":
        t = qt_step_back(query, provider)
        info["step_back"] = t
        hits = retrieve(provider, t, store, k=k)
        answer = generate(provider, query, hits)
        return answer, hits, info
    else:  # decompose
        subs = qt_decompose(query, provider, n=3)
        info["sub_questions"] = subs
        all_hits = []
        for s in subs:
            all_hits += retrieve(provider, s, store, k=max(1, k//len(subs) or 1))
        seen = set(); dedup = []
        for h in all_hits:
            if h["text"] not in seen:
                dedup.append(h); seen.add(h["text"])
        answer = generate(provider, query, dedup[:k])
        return answer, dedup[:k], info

def adaptive(provider: str, query: str, store: SimpleVectorStore, k: int = 4, user_ctx: Optional[str] = None) -> Tuple[str, List[Dict], str]:
    qtype = classify_query(provider, query)
    if qtype == "Factual":
        t = qt_rewrite(query, provider)
        hits = retrieve(provider, t, store, k=k)
    elif qtype == "Analytical":
        subs = qt_decompose(query, provider, n=3)
        hits = []
        for s in subs:
            hits += retrieve(provider, s, store, k=max(1, k//len(subs) or 1))
        seen=set(); dedup=[]
        for h in hits:
            if h["text"] not in seen:
                dedup.append(h); seen.add(h["text"])
        hits = dedup[:k]
    elif qtype == "Opinion":
        angles = qt_decompose(f"Suggest 3 distinct viewpoints on: {query}", provider, n=3)
        hits = []
        for a in angles:
            hits += retrieve(provider, f"{query} {a}", store, k=1)
    else:
        ctx_q = call_openai(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0) if provider=="openai" else call_gemini(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0)
        hits = retrieve(provider, ctx_q, store, k=k)
    answer = generate(provider, query, hits)
    return answer, hits, qtype

def run_pipeline(provider: str, pipeline: str, query: str, store: SimpleVectorStore, top_k: int, user_ctx: Optional[str], qt_mode: str):
    if pipeline == "Context-Enriched RAG":
        ans, hits = context_enriched(provider, query, store, k=top_k)
        meta = {"mode": "context_enriched"}
    elif pipeline == "Query Transform RAG":
        ans, hits, info = query_transform(provider, query, store, k=top_k, mode=qt_mode)
        meta = {"mode": f"query_transform/{qt_mode}", **info}
    else:
        ans, hits, qtype = adaptive(provider, query, store, k=top_k, user_ctx=user_ctx or None)
        meta = {"mode": "adaptive", "query_type": qtype}
    return ans, hits, meta

# --- Run ---
if go:
    if not query.strip():
        st.warning("Enter a question.")
        st.stop()
    if not (st.session_state.store_openai or st.session_state.store_gemini):
        st.error("Build or load at least one index (OpenAI or Gemini).")
        st.stop()

    col1, col2 = st.columns(2)

    # OpenAI side
    with col1:
        st.subheader("OpenAI RAG")
        if st.session_state.store_openai:
            ans_oai, hits_oai, meta_oai = run_pipeline("openai", pipeline, query, st.session_state.store_openai, top_k, user_ctx, qt_mode)
            time.sleep(pause_s)
            st.write(ans_oai if ans_oai else "—")
            ctx_oai = "\n".join([h["text"] for h in hits_oai]) if hits_oai else ""
            score_oai = score_answer(ans_oai or "", reference, ctx_oai)
            st.markdown(f"**Similarity score:** `{score_oai:.3f}`")
            if meta_oai:
                st.caption(f"Meta: {meta_oai}")
            with st.expander("Top passages (OpenAI index)"):
                for i, h in enumerate(hits_oai, 1):
                    meta = h.get("metadata", {})
                    st.markdown(f"**{i}.** (sim={h['similarity']:.3f}) — {meta.get('source','?')}#{meta.get('chunk_id','?')}")
                    st.code(h["text"][:1200], language="markdown")
        else:
            st.info("No OpenAI index loaded.")

    # Gemini side
    with col2:
        st.subheader("Gemini RAG")
        if st.session_state.store_gemini:
            ans_gem, hits_gem, meta_gem = run_pipeline("gemini", pipeline, query, st.session_state.store_gemini, top_k, user_ctx, qt_mode)
            time.sleep(pause_s)
            st.write(ans_gem if ans_gem else "—")
            ctx_gem = "\n".join([h["text"] for h in hits_gem]) if hits_gem else ""
            score_gem = score_answer(ans_gem or "", reference, ctx_gem)
            st.markdown(f"**Similarity score:** `{score_gem:.3f}`")
            if meta_gem:
                st.caption(f"Meta: {meta_gem}")
            with st.expander("Top passages (Gemini index)"):
                for i, h in enumerate(hits_gem, 1):
                    meta = h.get("metadata", {})
                    st.markdown(f"**{i}.** (sim={h['similarity']:.3f}) — {meta.get('source','?')}#{meta.get('chunk_id','?')}")
                    st.code(h["text"][:1200], language="markdown")
        else:
            st.info("No Gemini index loaded.")

    # Optional cross-answer similarity (OpenAI space)
    if oai_client and st.session_state.store_openai and st.session_state.store_gemini:
        try:
            if ans_oai and ans_gem:
                a_emb = np.array(embed_openai_one(ans_oai), dtype=np.float32)
                b_emb = np.array(embed_openai_one(ans_gem), dtype=np.float32)
                side_sim = float(np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-9))
                st.info(f"Answer-to-answer similarity (OpenAI embedding): {side_sim:.3f}")
        except Exception:
            pass

st.markdown("---")
st.caption("Built from your notebooks: context-enriched (G04/04), query-transform (G07/07), adaptive (G12/12) — unified into one app. Now with Prompt Guide loader.")