import os
import io
import re
import time
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
st.set_page_config(page_title="RAG: OpenAI vs Gemini", page_icon="🔬", layout="wide")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("Missing OPENAI_API_KEY in .env")
if not GEMINI_API_KEY:
    st.warning("Missing GEMINI_API_KEY in .env")

# Initialize clients if keys present
oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
STORE_OPENAI = os.path.join(DATA_DIR, "vector_store_openai.pkl")
STORE_GEMINI = os.path.join(DATA_DIR, "vector_store_gemini.pkl")


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
        except openai.RateLimitError as e:
            wait_s = min(60, 6 * (attempt + 1))
            st.info(f"[OpenAI embed] Rate limit; waiting {wait_s}s…")
            time.sleep(wait_s)
        except Exception as e:
            st.error(f"[OpenAI embed] {e}")
            break
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
    # Some versions return {'embedding': {'values': [...]}}
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list):  # older wrapper shape
            return emb
    # Fallback: try attribute style
    try:
        return resp.embedding.values  # type: ignore
    except Exception:
        pass
    # Last resort
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
            st.error(f"[Gemini embed] {e}")
            break
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
# LLM Calls (generation)
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
        except openai.RateLimitError as e:
            wait_s = min(90, 8 * (attempt + 1))
            st.info(f"[OpenAI gen] Rate limit; waiting {wait_s}s…")
            time.sleep(wait_s)
        except Exception as e:
            st.error(f"[OpenAI gen] {e}")
            break
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
            st.error(f"[Gemini gen] {e}")
            break
    return "ERROR: generation failed."


# =========================
# Build Indexes
# =========================
def build_index_from_uploads(files: List[io.BytesIO], chunk_size=1000, overlap=200,
                             provider: str = "openai") -> SimpleVectorStore:
    store = SimpleVectorStore()
    for f in files:
        name = getattr(f, "name", "uploaded.pdf")
        text = extract_text_from_pdf_bytes(f.read())
        chunks = chunk_text(text, chunk_size, overlap)
        if provider == "openai":
            embs = embed_openai_many(chunks)
        else:
            embs = embed_gemini_many(chunks)
        for i, (ch, emb) in enumerate(zip(chunks, embs)):
            if ch.strip():
                store.add_item(ch, emb, {"source": name, "chunk_id": i})
    return store

def build_index_from_folder(folder: str, chunk_size=1000, overlap=200,
                            provider: str = "openai") -> SimpleVectorStore:
    store = SimpleVectorStore()
    if not os.path.isdir(folder):
        return store
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fname)
        text = extract_text_from_pdf_path(path)
        chunks = chunk_text(text, chunk_size, overlap)
        if provider == "openai":
            embs = embed_openai_many(chunks)
        else:
            embs = embed_gemini_many(chunks)
        for i, (ch, emb) in enumerate(zip(chunks, embs)):
            if ch.strip():
                store.add_item(ch, emb, {"source": fname, "chunk_id": i})
    return store


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
# Retrieval + Answering
# =========================
def retrieve_and_answer_openai(query: str, store: SimpleVectorStore, k=4) -> Tuple[str, List[Dict]]:
    q_emb = embed_openai_one(query)
    hits = store.similarity_search(q_emb, k=k)
    context = "\n\n---\n\n".join([h["text"] for h in hits]) if hits else ""
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    ans = call_openai(prompt)
    return ans, hits

def retrieve_and_answer_gemini(query: str, store: SimpleVectorStore, k=4) -> Tuple[str, List[Dict]]:
    q_emb = embed_gemini_one(query)
    hits = store.similarity_search(q_emb, k=k)
    context = "\n\n---\n\n".join([h["text"] for h in hits]) if hits else ""
    prompt = f"You are a helpful assistant. Use only the provided context. If unsure, say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}"
    ans = call_gemini(prompt)
    return ans, hits


# =========================
# Scoring (cosine similarity)
# =========================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def eval_similarity(answer: str, ref_text: Optional[str] = None,
                    fallback_context: Optional[str] = None,
                    model: str = "text-embedding-3-small") -> float:
    """
    If ref_text is provided, score answer vs reference (OpenAI embeddings).
    Otherwise, score vs fallback_context (concatenated retrieved passages).
    """
    if not oai_client:
        return 0.0  # can't score without OpenAI eval model
    ans_emb = np.array(embed_openai_one(answer, model=model), dtype=np.float32)
    target = ref_text if (ref_text and ref_text.strip()) else (fallback_context or "")
    tgt_emb = np.array(embed_openai_one(target, model=model), dtype=np.float32)
    return cosine_sim(ans_emb, tgt_emb)


# =========================
# UI
# =========================
st.title("🔬 RAG Comparison: OpenAI vs Gemini")

with st.sidebar:
    st.header("Indexing")
    chunk_size = st.number_input("Chunk size", 200, 4000, 1000, 100)
    overlap = st.number_input("Overlap", 0, 1000, 200, 50)
    folder_path = st.text_input("Folder of PDFs (optional)", value=os.path.join(DATA_DIR, "pdfs"))
    uploads = st.file_uploader("Or upload PDFs", type=["pdf"], accept_multiple_files=True)
    build_oai = st.button("Build/Replace OpenAI Index")
    build_gem = st.button("Build/Replace Gemini Index")
    st.caption("Pro-tip: build both to compare apples-to-apples.")

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
st.header("Ask a Question")

query = st.text_input("Your question")
reference = st.text_area("(Optional) Reference answer for scoring", height=120)
top_k = st.slider("Top-K passages", 1, 10, 4)
pause_s = st.slider("Pause between API calls (sec) — increase if rate-limited", 0, 20, 6)
go = st.button("Run Comparison")

if go:
    if not query.strip():
        st.warning("Enter a question.")
        st.stop()

    # Require at least one store
    if not (st.session_state.store_openai or st.session_state.store_gemini):
        st.error("Build or load at least one index (OpenAI or Gemini).")
        st.stop()

    col1, col2 = st.columns(2)

    # --- OpenAI side ---
    with col1:
        st.subheader("OpenAI RAG")
        if st.session_state.store_openai:
            ans_oai, hits_oai = retrieve_and_answer_openai(query, st.session_state.store_openai, k=top_k)
            time.sleep(pause_s)
            st.write(ans_oai)

            # scoring (vs reference, else vs context)
            ctx_oai = "\n".join([h["text"] for h in hits_oai]) if hits_oai else ""
            score_oai = eval_similarity(ans_oai, reference, ctx_oai)
            st.markdown(f"**Similarity score:** `{score_oai:.3f}`")
            with st.expander("Top passages (OpenAI index)"):
                for i, h in enumerate(hits_oai, 1):
                    meta = h.get("metadata", {})
                    st.markdown(f"**{i}.** (sim={h['similarity']:.3f}) — {meta.get('source','?')}#{meta.get('chunk_id','?')}")
                    st.code(h["text"][:1200], language="markdown")
        else:
            st.info("No OpenAI index loaded.")

    # --- Gemini side ---
    with col2:
        st.subheader("Gemini RAG")
        if st.session_state.store_gemini:
            ans_gem, hits_gem = retrieve_and_answer_gemini(query, st.session_state.store_gemini, k=top_k)
            time.sleep(pause_s)
            st.write(ans_gem)

            # scoring (vs reference, else vs context)
            ctx_gem = "\n".join([h["text"] for h in hits_gem]) if hits_gem else ""
            score_gem = eval_similarity(ans_gem, reference, ctx_gem)
            st.markdown(f"**Similarity score:** `{score_gem:.3f}`")
            with st.expander("Top passages (Gemini index)"):
                for i, h in enumerate(hits_gem, 1):
                    meta = h.get("metadata", {})
                    st.markdown(f"**{i}.** (sim={h['similarity']:.3f}) — {meta.get('source','?')}#{meta.get('chunk_id','?')}")
                    st.code(h["text"][:1200], language="markdown")
        else:
            st.info("No Gemini index loaded.")

    # Quick comparison footer
    if oai_client:
        # Compare OpenAI answer vs Gemini answer (optional)
        if st.session_state.store_openai and st.session_state.store_gemini:
            try:
                oai_ans_emb = np.array(embed_openai_one(ans_oai), dtype=np.float32)
                gem_ans_emb = np.array(embed_openai_one(ans_gem), dtype=np.float32)
                side_sim = cosine_sim(oai_ans_emb, gem_ans_emb)
                st.info(f"Answer-to-Answer similarity (OpenAI embedding space): {side_sim:.3f}")
            except Exception:
                pass
