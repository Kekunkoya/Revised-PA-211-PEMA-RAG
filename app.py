import os
import io
import json
import time
import random
import pickle
from typing import List, Tuple, Optional, Dict

import numpy as np
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF

# =========================
# App setup
# =========================
st.set_page_config(page_title="PA 211 RAG — Repo PDFs (OpenAI vs Gemini vs Hybrid)", page_icon="📚", layout="wide")
load_dotenv()

# Secrets / env only (no widgets)
OPENAI_API_KEY = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in secrets or environment.")
if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY not found in secrets or environment.")

# Writable data dir (for general caches)
DATA_DIR = os.environ.get("STREAMLIT_DATA_DIR", "/mount/data")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)

# Default repo folders (you can change these)
DEFAULT_PDF_DIR = os.environ.get("PDF_DIR", "./pdfs")
DEFAULT_VECTOR_DIR = os.environ.get("VECTOR_DIR", "./vectors")

# =========================
# Session state init
# =========================
for key, default in [
    ("query", ""),
    ("reference", ""),
    ("prompt_guide", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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

# =========================
# PDF & chunking
# =========================
def list_pdfs(pdf_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(root, f))
    return sorted(paths)

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

def pdfs_to_texts_from_dir(pdf_dir: str, chunk_size=1000, overlap=200) -> Tuple[List[str], str]:
    pdf_paths = list_pdfs(pdf_dir)
    texts = []
    for path in pdf_paths:
        try:
            raw = extract_text_from_pdf_path(path)
            chunks = chunk_text(raw, chunk_size, overlap)
            for i, ch in enumerate(chunks):
                if ch.strip():
                    texts.append(f"[{os.path.basename(path)}#chunk_{i}]\n{ch}")
        except Exception as e:
            st.warning(f"Failed to read {path}: {e}")
    return texts, f"pdfs_{os.path.basename(os.path.abspath(pdf_dir))}_{chunk_size}_{overlap}"

# =========================
# Embeddings (OpenAI & Gemini)
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
        except Exception:
            time.sleep(retry_wait(attempt))
    # Fallback vector size for text-embedding-3-small
    return np.zeros((1536,), dtype=np.float32)

def embed_openai_many(texts: List[str], pause_s: int = 0, model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    embs = []
    for i, t in enumerate(texts):
        embs.append(embed_openai_one(t, model=model))
        if pause_s and (i + 1) % 15 == 0:
            time.sleep(pause_s)  # rate-limit helper
    return np.vstack(embs)

def parse_gemini_embed(resp) -> List[float]:
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

def embed_gemini_one(text: str, model: str = "models/embedding-001") -> np.ndarray:
    ensure_api_key("Gemini", GEMINI_API_KEY)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    for attempt in range(6):
        try:
            resp = genai.embed_content(model=model, content=text)
            vec = parse_gemini_embed(resp)
            return np.array(vec, dtype=np.float32)
        except Exception:
            time.sleep(retry_wait(attempt))
    return np.zeros((768,), dtype=np.float32)

def embed_gemini_many(texts: List[str], pause_s: int = 0, model: str = "models/embedding-001") -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    embs = []
    for i, t in enumerate(texts):
        embs.append(embed_gemini_one(t, model=model))
        if pause_s and (i + 1) % 15 == 0:
            time.sleep(pause_s)
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
        except Exception:
            time.sleep(retry_wait(attempt))
    return "ERROR: generation failed."

# =========================
# Vector I/O (prebuilt pickles)
# =========================
def vector_file_for(provider: str, vector_dir: str) -> str:
    os.makedirs(vector_dir, exist_ok=True)
    base = "openai_vectors.pkl" if provider == "OpenAI" else "gemini_vectors.pkl"
    return os.path.join(vector_dir, base)

def save_vectors(provider: str, vector_dir: str, embeddings: np.ndarray, texts: List[str]):
    path = vector_file_for(provider, vector_dir)
    with open(path, "wb") as f:
        pickle.dump({
            "provider": provider,
            "model": "text-embedding-3-small" if provider == "OpenAI" else "models/embedding-001",
            "embeddings": embeddings,
            "texts": texts
        }, f)

def load_vectors(provider: str, vector_dir: str) -> Optional[Dict]:
    path = vector_file_for(provider, vector_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.get("provider") == provider and "embeddings" in obj and "texts" in obj:
            return obj
    except Exception as e:
        st.warning(f"Failed to load vectors from {path}: {e}")
    return None

# =========================
# Retrieval + pipelines
# =========================
def retrieve_similar(query: str, embeddings: np.ndarray, texts: List[str], provider: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_emb = embed_openai_one(query) if provider == "OpenAI" else embed_gemini_one(query)
    norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-9) + 1e-9
    sims = (embeddings @ q_emb) / norms
    order = np.argsort(sims)[::-1][:k]
    return order, sims[order]

def context_enriched_answer(provider: str, query: str, top_texts: List[str]) -> str:
    joined = "\n\n".join(top_texts)
    brief_prompt = f"Create a concise brief (5-8 bullets) of key facts to answer.\n\nQuestion: {query}\n\nContext:\n{joined}\n\nBrief:"
    brief = call_openai(brief_prompt) if provider == "OpenAI" else call_gemini(brief_prompt)
    final_ctx = f"Brief:\n{brief}\n\n---\n\n{joined}"
    prompt = f"Answer using only this context.\n\nContext:\n{final_ctx}\n\nQuestion: {query}\nAnswer:"
    return call_openai(prompt) if provider == "OpenAI" else call_gemini(prompt)

def qt_rewrite(provider: str, query: str) -> str:
    p = f"Rewrite this query to be clearer and specific. Keep it short.\nQuery: {query}"
    return call_openai(p) if provider == "OpenAI" else call_gemini(p)

def qt_step_back(provider: str, query: str) -> str:
    p = f"Produce a single step-back question (the higher-level question that helps answer the original).\nOriginal: {query}\nStep-back:"
    return call_openai(p) if provider == "OpenAI" else call_gemini(p)

def qt_decompose(provider: str, query: str, n: int = 3) -> List[str]:
    p = f"Break the query into {n} short sub-questions (one per line).\nQuery: {query}\nSub-questions:"
    raw = call_openai(p) if provider == "OpenAI" else call_gemini(p)
    subs = [s.strip("- ").strip() for s in raw.splitlines() if s.strip()]
    return subs[:n] if subs else [query]

def query_transform_answer(provider: str, query: str, embeddings: np.ndarray, texts: List[str], k: int, mode: str):
    if mode == "rewrite":
        t = qt_rewrite(provider, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, provider, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if provider == "OpenAI" else call_gemini(prompt)
        return ans, {"transformed": t}, ctx
    elif mode == "step_back":
        t = qt_step_back(provider, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, provider, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if provider == "OpenAI" else call_gemini(prompt)
        return ans, {"step_back": t}, ctx
    else:
        subs = qt_decompose(provider, query, n=min(3, k))
        per = max(1, k // max(1, len(subs)))
        all_idxs = []
        for s in subs:
            idxs, _ = retrieve_similar(s, embeddings, texts, provider, per)
            all_idxs.extend(list(idxs))
        seen=set(); uniq=[]
        for i in all_idxs:
            if i not in seen:
                uniq.append(i); seen.add(i)
        ctx = "\n\n".join([texts[i] for i in uniq[:k]])
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans = call_openai(prompt) if provider == "OpenAI" else call_gemini(prompt)
        return ans, {"sub_questions": subs}, ctx

def classify_query(provider: str, query: str) -> str:
    p = f"""Classify the query into exactly one category:
- Factual
- Analytical
- Opinion
- Contextual

Query: {query}
Return ONLY the category name.
"""
    out = call_openai(p) if provider == "OpenAI" else call_gemini(p)
    for c in ["Factual", "Analytical", "Opinion", "Contextual"]:
        if c.lower() in (out or "").lower():
            return c
    return "Factual"

def adaptive_answer(provider: str, query: str, embeddings: np.ndarray, texts: List[str], k: int, user_ctx: str):
    qtype = classify_query(provider, query)
    if qtype == "Factual":
        t = qt_rewrite(provider, query)
        idxs, _ = retrieve_similar(t, embeddings, texts, provider, k)
        ctx = "\n\n".join([texts[i] for i in idxs])
    elif qtype == "Analytical":
        subs = qt_decompose(provider, query, n=min(3, k))
        per = max(1, k // max(1, len(subs)))
        all_idxs = []
        for s in subs:
            idxs, _ = retrieve_similar(s, embeddings, texts, provider, per)
            all_idxs.extend(list(idxs))
        seen=set(); uniq=[]
        for i in all_idxs:
            if i not in seen:
                uniq.append(i); seen.add(i)
        ctx = "\n\n".join([texts[i] for i in uniq[:k]])
    elif qtype == "Opinion":
        angles = qt_decompose(provider, f"Suggest 3 distinct viewpoints on: {query}", n=3)
        all_idxs = []
        for a in angles:
            idxs, _ = retrieve_similar(f"{query} {a}", embeddings, texts, provider, 1)
            all_idxs.extend(list(idxs))
        ctx = "\n\n".join([texts[i] for i in all_idxs[:k]])
    else:  # Contextual
        reform = call_openai(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0) if provider=="OpenAI" else call_gemini(f"Reformulate with context.\nQuery: {query}\nContext: {user_ctx}", temperature=0)
        idxs, _ = retrieve_similar(reform, embeddings, texts, provider, k)
        ctx = "\n\n".join([texts[i] for i in idxs])

    prompt = f"Answer using only the context.\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    ans = call_openai(prompt) if provider == "OpenAI" else call_gemini(prompt)
    return ans, qtype, ctx

def combined_pipeline(provider: str, query: str, embeddings: np.ndarray, texts: List[str], k: int, user_ctx: str):
    """Context-Enriched → Query-Transform (rewrite+decompose) → Adaptive-style consolidation."""
    # 1) Context-enriched seed
    idxs, _ = retrieve_similar(query, embeddings, texts, provider, min(k, 4))
    seed_ctx = [texts[i] for i in idxs]

    # 2) Query transform (rewrite + decompose)
    rewrite = qt_rewrite(provider, query)
    subs = qt_decompose(provider, query, n=min(3, k))
    # gather contexts
    ctx_parts = []
    if seed_ctx:
        ctx_parts.extend(seed_ctx)
    # rewrite
    r_idx, _ = retrieve_similar(rewrite, embeddings, texts, provider, min(k, 4))
    ctx_parts.extend([texts[i] for i in r_idx])
    # decompose
    per = max(1, k // max(1, len(subs)))
    all_idxs = []
    for s in subs:
        ix, _ = retrieve_similar(s, embeddings, texts, provider, per)
        all_idxs.extend(list(ix))
    seen=set(); uniq=[]
    for i in all_idxs:
        if i not in seen:
            uniq.append(i); seen.add(i)
    ctx_parts.extend([texts[i] for i in uniq[:k]])

    # dedupe & cut
    merged = []
    seen_txt = set()
    for t in ctx_parts:
        key = t[:120]  # cheap dedupe
        if key not in seen_txt:
            merged.append(t)
            seen_txt.add(key)
    merged = merged[:max(k, 6)]

    # 3) Adaptive classification + final answer
    qtype = classify_query(provider, query)
    final_ctx = "\n\n".join(merged)
    prompt = f"Answer using only the context.\n\nContext:\n{final_ctx}\n\nQuestion: {query}\nAnswer:"
    ans = call_openai(prompt) if provider == "OpenAI" else call_gemini(prompt)
    return ans, qtype, final_ctx

# =========================
# Scoring
# =========================
def score_answer(answer: str, fallback_context: str) -> float:
    # Score in OpenAI embedding space vs reference if provided else vs context
    if not OPENAI_API_KEY:
        return 0.0
    try:
        ans_emb = embed_openai_one(answer)
        tgt = (st.session_state.get("reference") or "").strip() or fallback_context
        tgt_emb = embed_openai_one(tgt)
        return cosine_sim(ans_emb, tgt_emb)
    except Exception:
        return 0.0

def cross_answer_similarity(ans_a: str, ans_b: str) -> float:
    if not OPENAI_API_KEY:
        return 0.0
    try:
        a = embed_openai_one(ans_a)
        b = embed_openai_one(ans_b)
        return cosine_sim(a, b)
    except Exception:
        return 0.0

# =========================
# Sidebar controls (no API widgets)
# =========================
with st.sidebar:
    st.header("Sources (Repo Paths)")
    pdf_dir = st.text_input("PDF folder (repo)", value=DEFAULT_PDF_DIR)
    vector_dir = st.text_input("Vectors folder (repo)", value=DEFAULT_VECTOR_DIR)
    rebuild_vectors = st.checkbox("Rebuild vectors even if found", value=False)

    st.header("Chunking")
    chunk_size = st.number_input("Chunk size", 200, 4000, 1000, 100)
    overlap = st.number_input("Overlap", 0, 1000, 200, 50)

    st.header("Rate-limit helper")
    pause_s = st.slider("Pause between batches (seconds)", 0, 20, 6)

# =========================
# Main UI
# =========================
st.title("📚 PA 211 RAG — Repo PDFs (OpenAI vs Gemini vs Hybrid)")
st.caption("Modes: Context-Enriched • Query-Transform • Adaptive • Combined (context+transform+adaptive). Uses repo folders for PDFs and vectors.")

col_q, col_r = st.columns([2, 1])
with col_q:
    query = st.text_area("Your question", key="query", height=120)
with col_r:
    st.markdown("*(Optional) Reference for scoring*")
    reference = st.text_area("Reference answer", key="reference", height=120)

col_top = st.columns(3)
with col_top[0]:
    mode = st.selectbox("RAG Mode", ["Context-Enriched", "Query-Transform", "Adaptive", "Combined"])
with col_top[1]:
    provider_choice = st.selectbox("Provider", ["OpenAI", "Gemini", "Both"])
with col_top[2]:
    top_k = st.slider("Top-K retrieved", 1, 10, 4)

user_ctx = st.text_input("User context (for Adaptive/Contextual)")

run_btn = st.button("Run")

# =========================
# Build or load vectors
# =========================
def build_or_load(provider: str) -> Tuple[np.ndarray, List[str], str]:
    # Try to load vectors first unless forced rebuild
    if not rebuild_vectors:
        loaded = load_vectors(provider, vector_dir)
        if loaded and isinstance(loaded.get("embeddings"), np.ndarray) and isinstance(loaded.get("texts"), list):
            return loaded["embeddings"], loaded["texts"], "loaded"

    # Build from PDFs in repo
    texts, source_name = pdfs_to_texts_from_dir(pdf_dir, chunk_size, overlap)
    st.info(f"Found {len(texts)} text chunks from PDFs in '{pdf_dir}'. Building {provider} embeddings…")
    if provider == "OpenAI":
        embs = embed_openai_many(texts, pause_s=pause_s)
    else:
        embs = embed_gemini_many(texts, pause_s=pause_s)

    save_vectors(provider, vector_dir, embs, texts)
    st.success(f"Saved vectors to {vector_file_for(provider, vector_dir)}")
    return embs, texts, "built"

def run_pipeline(provider: str, embeddings: np.ndarray, texts: List[str]) -> Dict[str, str]:
    if mode == "Context-Enriched":
        idxs, _ = retrieve_similar(query, embeddings, texts, provider, top_k)
        picked = [texts[i] for i in idxs]
        ans = context_enriched_answer(provider, query, picked)
        ctx = "\n\n".join(picked)
    elif mode == "Query-Transform":
        # default to rewrite mode inside transform (or you could add a selector)
        ans, info, ctx = query_transform_answer(provider, query, embeddings, texts, top_k, mode="rewrite")
    elif mode == "Adaptive":
        ans, qtype, ctx = adaptive_answer(provider, query, embeddings, texts, top_k, user_ctx)
    else:  # Combined
        ans, qtype, ctx = combined_pipeline(provider, query, embeddings, texts, top_k, user_ctx)

    score = score_answer(ans or "", ctx or "")
    return {"answer": ans, "context": ctx, "score": f"{score:.3f}"}

# =========================
# Run
# =========================
if run_btn:
    if not os.path.isdir(pdf_dir):
        st.error(f"PDF folder does not exist: {pdf_dir}")
    else:
        if provider_choice in ["OpenAI", "Both"]:
            oai_embs, oai_texts, oai_status = build_or_load("OpenAI")
        if provider_choice in ["Gemini", "Both"]:
            gem_embs, gem_texts, gem_status = build_or_load("Gemini")

        if provider_choice == "OpenAI":
            st.subheader("OpenAI")
            res = run_pipeline("OpenAI", oai_embs, oai_texts)
            st.write(res["answer"])
            st.markdown(f"**Similarity score:** `{res['score']}`")
            with st.expander("Context used"):
                st.code(res["context"][:4000], language="markdown")

        elif provider_choice == "Gemini":
            st.subheader("Gemini")
            res = run_pipeline("Gemini", gem_embs, gem_texts)
            st.write(res["answer"])
            st.markdown(f"**Similarity score:** `{res['score']}`")
            with st.expander("Context used"):
                st.code(res["context"][:4000], language="markdown")

        else:  # Both
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("OpenAI")
                res_o = run_pipeline("OpenAI", oai_embs, oai_texts)
                st.write(res_o["answer"])
                st.markdown(f"**Similarity score:** `{res_o['score']}`")
                with st.expander("Context used"):
                    st.code(res_o["context"][:4000], language="markdown")

            time.sleep(max(0, pause_s // 2))  # small pause

            with c2:
                st.subheader("Gemini")
                res_g = run_pipeline("Gemini", gem_embs, gem_texts)
                st.write(res_g["answer"])
                st.markdown(f"**Similarity score:** `{res_g['score']}`")
                with st.expander("Context used"):
                    st.code(res_g["context"][:4000], language="markdown")

            # Cross-model similarity and Hybrid consensus
            st.markdown("---")
            if res_o.get("answer") and res_g.get("answer"):
                sim_ab = cross_answer_similarity(res_o["answer"], res_g["answer"])
                st.markdown(f"**OpenAI ↔ Gemini Answer Similarity:** `{sim_ab:.3f}`")

                # Hybrid consensus (uses whichever key you have, prefers OpenAI)
                consensus_ctx = (res_o["context"] + "\n\n" + res_g["context"])[:8000]
                consensus_prompt = f"Combine the evidence from both contexts and produce a single, concise answer.\n\nContext A:\n{res_o['context']}\n\nContext B:\n{res_g['context']}\n\nQuestion: {query}\nAnswer:"
                if OPENAI_API_KEY:
                    hybrid_ans = call_openai(consensus_prompt)
                elif GEMINI_API_KEY:
                    hybrid_ans = call_gemini(consensus_prompt)
                else:
                    hybrid_ans = "No API keys available for hybrid synthesis."

                hy_score = score_answer(hybrid_ans or "", consensus_ctx or "")
                st.subheader("Hybrid (Consensus)")
                st.write(hybrid_ans)
                st.markdown(f"**Similarity score:** `{hy_score:.3f}`")
                with st.expander("Hybrid context used"):
                    st.code(consensus_ctx[:4000], language="markdown")

# Footer
st.markdown("---")
st.caption("Reads PDFs and (optionally) prebuilt vector pickles from repo folders. No API key widgets; uses Streamlit Secrets / env only.")
