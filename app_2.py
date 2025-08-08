# -*- coding: utf-8 -*-
import os
import time
import json
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st

# -------------------------
# Page config + Intro tab content
# -------------------------
st.set_page_config(page_title="PA 211 / PEMA RAG - OpenAI vs Gemini vs Both", layout="wide")

INTRO_MD = """
# PA 211 / PEMA RAG — Demo Guide

This demo answers questions using RAG (Retrieval-Augmented Generation) over the PDFs in the repo root and optionally an auxiliary QA memory file (PA211_expanded_dataset.json). It lets you compare OpenAI, Gemini, or a merged Both view.

---

## What's in this demo?

- PDF ingestion: All .pdf files at the repo root are loaded and chunked.
- Two vector stores: One per model family (OpenAI and Gemini). Vectors are saved as .pkl files so subsequent runs are fast.
- Multiple retrieval modes (pipelines):
  - Standard: vanilla chunk retrieval (fast, precise).
  - Contextual: context-enriched chunks (each chunk sees its neighbors) -> better recall.
  - Query Transformation: rewrite / step-back / decompose, then retrieve (recall boost; slightly slower).
  - Adaptive: heuristics classify the query (Factual / Analytical / Opinion / Contextual) to decide which strategy to apply.
  - Combined: merge Standard + Contextual + (optional transforms) for a strong, balanced top-K set.
- QA memory (optional): A small side index built from PA211_expanded_dataset.json. Its top hint can be added to the prompt (as a nudge), while the final answer must still be grounded in PDFs.
- Comparison: Run OpenAI, Gemini, Both (merged) and see answers side-by-side with retrieval scores.

---

## End-to-end workflow

PDFs at repo root -> extract & chunk -> (standard/contextual chunks)
Optionally: build QA memory from PA211_expanded_dataset.json

Embed with OpenAI + Gemini -> save as .pkl -> vector stores

Retrieval pipelines (Standard / Contextual / Query-Transform / Adaptive / Combined)
-> retrieve top-K (+ optional QA hint) -> build grounded prompt
-> LLM (OpenAI / Gemini / Both) -> Answer + similarity metrics

---

## Key ideas

- Chunking: PDFs are split into overlapping chunks (default ~900 chars with 250 overlap).
- Contextual chunks: Each chunk is expanded with its neighbors to boost recall for cross-page topics.
- Two independent vector stores: OpenAI embeddings power OpenAI retrieval; Gemini embeddings power Gemini retrieval. This isolates model behavior fairly.
- QA memory: Used as a hint (optional) to nudge the model; final answer must still come from the PDFs.
- Similarity scores: Quick sanity checks of how well the answer aligns with the retrieved evidence.

---

## Retrieval modes (pipelines)

- Standard
  Single query -> retrieve top-K from standard chunks -> answer. Use when you want fast, precise lookups.

- Contextual
  Single query -> retrieve top-K from contextual chunks -> answer. Use when answers likely span multiple pages/sections.

- Query Transformation
  Query -> rewrite / step-back / decompose -> retrieve for each -> merge top-K -> answer.
  Use for ambiguous or complex questions; costs a few extra LLM calls.

- Adaptive
  Heuristic classify (Factual / Analytical / Opinion / Contextual) -> choose a strategy (rewrite, decompose, etc.) and pull from both standard + contextual stores -> answer.
  Good default when you don't know the shape of the question.

- Combined
  Merge Standard + Contextual results; optional transforms. Often robust with reasonable cost.

---

## Compare OpenAI vs Gemini vs Both

- OpenAI: uses OpenAI embeddings + OpenAI generation.
- Gemini: uses Gemini embeddings + Gemini generation.
- Both (merged): merges contexts from both stores and synthesizes a single answer (uses OpenAI if available, else Gemini).

Why compare?
Different embedding spaces and LLMs will surface different passages and phrasing. Side-by-side can reveal coverage gaps or strengths.

---

## Performance notes

- Prebuilt vectors: First build can take minutes; after that, it is instant. Commit the .pkl files with the repo for demos.
- Gemini embedding is per-text; OpenAI supports batching. That is why OpenAI builds usually run faster.
- Use Contextual or Combined for better recall; start with Standard if you need speed.
- Turn off QA memory if you want purely PDF-grounded behavior with zero extra retrieval.

---

## Troubleshooting

- "I don't have enough info." -> Try Contextual or Combined, bump Top-K, or enable Query Transformation.
- Empty or low similarity -> The answer likely is not in the PDFs; rephrase the question, or confirm the document actually contains it.
- Slow first run -> Build/Refresh vectors once, then commit .pkl to the repo.
"""

# Create tabs ONCE
tabs = st.tabs(["Introduction", "RAG Demo"])

with tabs[0]:
    st.markdown(INTRO_MD)

# -------------------------
# Config and secrets
# -------------------------
def list_repo_pdfs() -> List[str]:
    return sorted([f for f in os.listdir(".") if f.lower().endswith(".pdf")])

PDF_FILES = list_repo_pdfs()
CHUNK_SIZE = 900
OVERLAP = 250
TOP_K_DEFAULT = 4

# Vector pickle names (at repo root)
PREBUILT = {
    ("OpenAI", "standard"):   "openai_vectors_std.pkl",
    ("OpenAI", "contextual"): "openai_vectors_ctx.pkl",
    ("OpenAI", "qa"):         "openai_vectors_qa.pkl",
    ("Gemini", "standard"):   "gemini_vectors_std.pkl",
    ("Gemini", "contextual"): "gemini_vectors_ctx.pkl",
    ("Gemini", "qa"):         "gemini_vectors_qa.pkl",
}

QA_DATA_FILE = "PA211_expanded_dataset.json"  # optional

# Secrets (no widgets)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    with tabs[1]:
        st.error("Both OPENAI_API_KEY and GEMINI_API_KEY are missing. Add at least one in Streamlit secrets.")
    st.stop()

# -------------------------
# Lazy imports for APIs
# -------------------------
def _lazy_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def _lazy_genai():
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# -------------------------
# Small utils
# -------------------------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def avg_sim(hits: List[dict]) -> float:
    if not hits: return 0.0
    return float(np.mean([h.get("similarity", 0.0) for h in hits]))

# -------------------------
# PDF -> Text
# -------------------------
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(path)
    except Exception:
        return ""
    parts = []
    for p in doc:
        try:
            parts.append(p.get_text("text"))
        except Exception:
            pass
    doc.close()
    return "\n".join(parts).strip()

@st.cache_data(show_spinner=False)
def load_all_pdfs_text(pdf_files: List[str]) -> Dict[str, str]:
    out = {}
    for p in pdf_files:
        out[p] = extract_text_from_pdf(p) if os.path.exists(p) else ""
    return out

def chunk_text(text: str, n: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    if not text: return []
    step = max(1, n - overlap)
    return [text[i:i + n] for i in range(0, len(text), step) if text[i:i + n].strip()]

def contextualize_chunks(chunks: List[str], window: int = 1) -> List[str]:
    if not chunks: return []
    out = []
    for i in range(len(chunks)):
        lo = max(0, i - window)
        hi = min(len(chunks), i + window + 1)
        out.append("\n\n".join(chunks[lo:hi]))
    return out

# -------------------------
# Embeddings
# -------------------------
def embed_openai_many(texts: List[str], batch_size: int = 64) -> np.ndarray:
    client = _lazy_openai()
    model = "text-embedding-3-small"
    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = [t[:3000] for t in texts[i:i+batch_size]]
        # Retry wrapper
        for _ in range(4):
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                for item in resp.data:
                    vecs.append(np.array(item.embedding, dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
        else:
            # fallback zeros for this batch
            for _ in chunk:
                vecs.append(np.zeros((1536,), dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 1536), dtype=np.float32)

def embed_gemini_many(texts: List[str]) -> np.ndarray:
    genai = _lazy_genai()
    model = "models/embedding-001"
    vecs = []
    for t in texts:
        t2 = t[:3000]
        ok = False
        for _ in range(4):
            try:
                r = genai.embed_content(model=model, content=t2)
                # google-generativeai can return dict or object; normalize
                if isinstance(r, dict):
                    vec = r.get("embedding")
                    if isinstance(vec, dict) and "values" in vec:
                        vec = vec["values"]
                else:
                    emb = getattr(r, "embedding", None)
                    vec = getattr(emb, "values", []) if emb is not None else []
                vecs.append(np.array(vec, dtype=np.float32))
                ok = True
                break
            except Exception:
                time.sleep(1)
        if not ok:
            vecs.append(np.zeros((768,), dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)

# -------------------------
# Vector Store
# -------------------------
class VectorStore:
    def __init__(self, provider: str, embeddings: np.ndarray, metas: List[dict]):
        self.provider = provider
        self.embeddings = embeddings
        self.metas = metas
        self.dim = embeddings.shape[1] if embeddings.size else 0

    def query_embed(self, text: str) -> np.ndarray:
        if self.provider == "OpenAI":
            return embed_openai_many([text])[0]
        else:
            return embed_gemini_many([text])[0]

    def search(self, query: str, k: int = 4) -> List[dict]:
        if not self.embeddings.size: return []
        q = self.query_embed(query)
        norms = (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q) + 1e-9)
        sims = (self.embeddings @ q) / norms
        order = np.argsort(-sims)[:k]
        out = []
        for idx in order:
            m = dict(self.metas[idx])
            m["similarity"] = float(sims[idx])
            m["row"] = idx
            out.append(m)
        return out

def prebuilt_path(provider: str, mode: str) -> str:
    return PREBUILT.get((provider, mode), "")

def load_prebuilt(provider: str, mode: str) -> Optional[VectorStore]:
    p = prebuilt_path(provider, mode)
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        embs = obj.get("embeddings")
        metas = obj.get("metas")
        if isinstance(embs, np.ndarray) and isinstance(metas, list):
            return VectorStore(provider, embs, metas)
    except Exception:
        pass
    return None

def save_prebuilt(provider: str, mode: str, store: VectorStore):
    p = prebuilt_path(provider, mode)
    if not p: return
    try:
        with open(p, "wb") as f:
            pickle.dump({"embeddings": store.embeddings, "metas": store.metas}, f)
    except Exception:
        pass

def build_corpus_from_pdfs(all_text: Dict[str, str], mode: str) -> Tuple[List[str], List[dict]]:
    texts, metas = [], []
    for fname, full in all_text.items():
        if not full:
            continue
        chunks = chunk_text(full)
        if mode == "contextual":
            chunks = contextualize_chunks(chunks, window=1)
        for i, ch in enumerate(chunks):
            wrapped = f"[SOURCE: {os.path.basename(fname)} | chunk {i}]\n{ch}"
            texts.append(wrapped)
            metas.append({"source": fname, "chunk_id": i, "text": wrapped})
    return texts, metas

def build_corpus_from_qa(qa_items: List[dict]) -> Tuple[List[str], List[dict]]:
    texts, metas = [], []
    for i, item in enumerate(qa_items):
        q = (item.get("question") or "").strip()
        a = (item.get("ideal_answer") or "").strip()
        if not q and not a:
            continue
        txt = f"[QA MEMORY] {q}\n{a}" if q else f"[QA MEMORY]\n{a}"
        texts.append(txt)
        metas.append({"qa_id": i, "text": txt})
    return texts, metas

@st.cache_resource(show_spinner=False)
def build_store(provider: str, mode: str, all_text: Dict[str, str], qa_items: Optional[List[dict]] = None) -> VectorStore:
    # 1) Try prebuilt
    pre = load_prebuilt(provider, mode)
    if pre:
        return pre

    # 2) Build
    if mode in ("standard", "contextual"):
        texts, metas = build_corpus_from_pdfs(all_text, mode=mode)
    elif mode == "qa":
        texts, metas = build_corpus_from_qa(qa_items or [])
    else:
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])

    if not texts:
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])

    embs = embed_openai_many(texts) if provider == "OpenAI" else embed_gemini_many(texts)
    store = VectorStore(provider, embs, metas)
    save_prebuilt(provider, mode, store)
    return store

def rebuild_and_save(provider: str, mode: str, all_text: Dict[str, str], qa_items: Optional[List[dict]] = None) -> VectorStore:
    if mode in ("standard", "contextual"):
        texts, metas = build_corpus_from_pdfs(all_text, mode=mode)
    elif mode == "qa":
        texts, metas = build_corpus_from_qa(qa_items or [])
    else:
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])

    if not texts:
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])
    embs = embed_openai_many(texts) if provider == "OpenAI" else embed_gemini_many(texts)
    store = VectorStore(provider, embs, metas)
    save_prebuilt(provider, mode, store)
    return store

# -------------------------
# LLM helpers
# -------------------------
def llm(engine: str, prompt: str, temperature: float = 0.1) -> str:
    if engine == "OpenAI":
        client = _lazy_openai()
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Answer strictly using the provided context."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception:
                time.sleep(1)
        return "ERROR"
    else:
        genai = _lazy_genai()
        mdl = genai.GenerativeModel("gemini-2.0-flash")
        for _ in range(3):
            try:
                r = mdl.generate_content(prompt)
                return (getattr(r, "text", "") or "").strip()
            except Exception:
                time.sleep(1)
        return "ERROR"

def transform_query(engine: str, query: str, ttype: str) -> List[str]:
    if ttype == "rewrite":
        p = f"Rewrite the query to be clearer and more specific.\nQuery: {query}\nRewritten:"
        return [llm(engine, p)]
    if ttype == "step_back":
        p = f"Give one higher-level step-back question that helps retrieve broader context.\nOriginal: {query}\nStep-back:"
        return [llm(engine, p)]
    if ttype == "decompose":
        p = f"Break the query into 3 short sub-questions, one per line.\nQuery: {query}\nSub-questions:"
        raw = llm(engine, p, 0.2)
        lines = [l.strip("-• ").strip() for l in raw.splitlines() if l.strip()]
        return lines[:3] if lines else [query]
    return [query]

def classify_query_rule(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["why", "how", "compare", "difference", "pros", "cons", "tradeoff"]):
        return "Analytical"
    if any(w in q for w in ["opinion", "should i", "what do you think"]):
        return "Opinion"
    if any(w in q for w in ["me", "my", "our", "we", "near me", "zip", "17104"]):
        return "Contextual"
    return "Factual"

# -------------------------
# Answer + scoring
# -------------------------
def build_prompt(query: str, pdf_hits: List[dict], qa_hint: Optional[str]) -> str:
    ctx = "\n\n---\n\n".join([h["text"] for h in pdf_hits]) if pdf_hits else "(no context)"
    hint = f"\n\n[QA MEMORY HINT]\n{qa_hint.strip()}" if qa_hint else ""
    return (
        "Use ONLY the PDF context below to answer. "
        "If it doesn't contain enough info, say you don't have enough info."
        f"\n\nContext:\n{ctx}{hint}\n\nQuestion: {query}\nAnswer:"
    )

def answer(engine: str, query: str, pdf_hits: List[dict], qa_hint: Optional[str]) -> str:
    return llm(engine, build_prompt(query, pdf_hits, qa_hint), 0.1)

def answer_similarity_against_hits(engine: str, answer_text: str, store: Optional[VectorStore], hits: List[dict]) -> Tuple[float, float]:
    """
    Returns (sim_to_centroid, max_sim_to_any_hit) using the engine's embedding model.
    If store is None (e.g., merged hits from both engines), re-embed hit texts directly.
    """
    if not hits or not answer_text.strip():
        return 0.0, 0.0

    if engine == "OpenAI":
        ans_vec = embed_openai_many([answer_text])[0]
        if store is not None and store.embeddings.size:
            rows = [h["row"] for h in hits]
            H = store.embeddings[rows]
        else:
            H = embed_openai_many([h["text"] for h in hits])
    else:
        ans_vec = embed_gemini_many([answer_text])[0]
        if store is not None and store.embeddings.size:
            rows = [h["row"] for h in hits]
            H = store.embeddings[rows]
        else:
            H = embed_gemini_many([h["text"] for h in hits])

    if H.size == 0:
        return 0.0, 0.0

    centroid = np.mean(H, axis=0)
    sim_centroid = _cos(ans_vec, centroid)
    sim_max = float(np.max([(ans_vec @ h) / (np.linalg.norm(ans_vec)*np.linalg.norm(h)+1e-9) for h in H]))
    return sim_centroid, sim_max

# -------------------------
# Retrieval pipelines
# -------------------------
def retrieve_standard(store: VectorStore, query: str, k: int) -> List[dict]:
    return store.search(query, k=k)

def retrieve_contextual(store_ctx: VectorStore, query: str, k: int) -> List[dict]:
    return store_ctx.search(query, k=k)

def retrieve_query_transform(engine: str, store_std: VectorStore, query: str, k: int) -> List[dict]:
    tqs = (transform_query(engine, query, "rewrite") +
           transform_query(engine, query, "step_back") +
           transform_query(engine, query, "decompose"))
    seen = set(); hits = []
    for tq in tqs:
        for h in store_std.search(tq, k=max(2, k // 2 + 1)):
            key = (h["source"], h["chunk_id"])
            if key not in seen:
                seen.add(key); hits.append(h)
    return sorted(hits, key=lambda x: -x["similarity"])[:k]

def retrieve_adaptive(engine: str, store_ctx: VectorStore, store_std: VectorStore, query: str, k: int) -> Tuple[str, List[dict]]:
    qtype = classify_query_rule(query)
    if qtype == "Factual":
        qs = transform_query(engine, query, "rewrite")
    elif qtype == "Analytical":
        qs = transform_query(engine, query, "decompose")
    elif qtype == "Opinion":
        qs = [query] + transform_query(engine, query, "step_back")
    else:  # Contextual
        qs = [f"{query} (shelter rules, local eligibility, resources)"]

    seen = set(); hits = []
    # mix contextual + standard for coverage
    for q in qs:
        for h in store_ctx.search(q, k=max(2, k//2)):
            key = (h["source"], h["chunk_id"])
            if key not in seen:
                seen.add(key); hits.append(h)
        for h in store_std.search(q, k=max(2, k//2)):
            key = (h["source"], h["chunk_id"])
            if key not in seen:
                seen.add(key); hits.append(h)
    hits = sorted(hits, key=lambda x: -x["similarity"])[:k]
    return qtype, hits

def retrieve_combined(engine: str, store_std: VectorStore, store_ctx: VectorStore, query: str, k: int, enable_transforms: bool) -> List[dict]:
    hits = store_std.search(query, k=k)
    existing = {(h["source"], h["chunk_id"]) for h in hits}
    for h in store_ctx.search(query, k=k):
        key = (h["source"], h["chunk_id"])
        if key not in existing:
            hits.append(h); existing.add(key)
    if enable_transforms:
        tqs = transform_query(engine, query, "rewrite") + transform_query(engine, query, "step_back")
        for tq in tqs:
            for h in store_std.search(tq, k=max(2, k//2)):
                key = (h["source"], h["chunk_id"])
                if key not in existing:
                    hits.append(h); existing.add(key)
    return sorted(hits, key=lambda x: -x["similarity"])[:k]

# -------------------------
# UI - Demo tab
# -------------------------
with tabs[1]:
    st.title("PA 211 / PEMA RAG — OpenAI vs Gemini vs Both (Repo PDFs)")

    with st.expander("PDF status", expanded=False):
        if not PDF_FILES:
            st.error("No PDFs found in the repo root.")
        else:
            missing = [p for p in PDF_FILES if not os.path.exists(p)]
            if missing:
                st.error("Missing files:\n" + "\n".join(missing))
            else:
                st.success("All PDFs detected.")
            st.write("Using these PDFs:")
            st.write(PDF_FILES)

    mode = st.selectbox(
        "Retrieval mode",
        ["Standard", "Contextual", "Query Transformation", "Adaptive", "Combined"],
        index=0
    )
    enable_transforms = st.checkbox("Enable transforms in Combined mode", value=False)
    use_qa_memory = st.checkbox("Use QA memory (PA211_expanded_dataset.json) for hints", value=True)
    compare_choice = st.radio(
        "Engine",
        ["OpenAI", "Gemini", "Compare: OpenAI vs Gemini", "Compare: OpenAI vs Gemini vs Both"],
        index=2
    )
    top_k = st.slider("Top-K passages", 1, 8, TOP_K_DEFAULT)

    default_q = "What does the Red Cross and PEMA shelter guide say about bringing pets to emergency shelters in Harrisburg?"
    query = st.text_area("Your question", value=default_q, height=80)

    # Load PDFs text once
    with st.spinner("Loading PDFs..."):
        ALL_TEXT = load_all_pdfs_text(PDF_FILES)

    # Load QA data if any
    qa_items: List[dict] = []
    if use_qa_memory and os.path.exists(QA_DATA_FILE):
        try:
            with open(QA_DATA_FILE, "r", encoding="utf-8") as f:
                qa_items = json.load(f)
        except Exception:
            qa_items = []

    # Build/Refresh buttons
    colb = st.columns(5)
    with colb[0]:
        rebuild_oai_std = st.button("Rebuild OpenAI (Standard)")
    with colb[1]:
        rebuild_oai_ctx = st.button("Rebuild OpenAI (Contextual)")
    with colb[2]:
        rebuild_gem_std = st.button("Rebuild Gemini (Standard)")
    with colb[3]:
        rebuild_gem_ctx = st.button("Rebuild Gemini (Contextual)")
    with colb[4]:
        rebuild_qa = st.button("Rebuild QA vectors")

    if rebuild_oai_std:
        with st.spinner("Rebuilding OpenAI standard vectors…"):
            _ = rebuild_and_save("OpenAI", "standard", ALL_TEXT)
            st.success("OpenAI (standard) rebuilt.")
    if rebuild_oai_ctx:
        with st.spinner("Rebuilding OpenAI contextual vectors…"):
            _ = rebuild_and_save("OpenAI", "contextual", ALL_TEXT)
            st.success("OpenAI (contextual) rebuilt.")
    if rebuild_gem_std:
        with st.spinner("Rebuilding Gemini standard vectors…"):
            _ = rebuild_and_save("Gemini", "standard", ALL_TEXT)
            st.success("Gemini (standard) rebuilt.")
    if rebuild_gem_ctx:
        with st.spinner("Rebuilding Gemini contextual vectors…"):
            _ = rebuild_and_save("Gemini", "contextual", ALL_TEXT)
            st.success("Gemini (contextual) rebuilt.")
    if rebuild_qa and qa_items:
        with st.spinner("Rebuilding QA vectors…"):
            _ = rebuild_and_save("OpenAI", "qa", {}, qa_items) if OPENAI_API_KEY else None
            _ = rebuild_and_save("Gemini", "qa", {}, qa_items) if GEMINI_API_KEY else None
            st.success("QA vectors rebuilt.")

    # Helpers to get stores
    def get_store(provider: str, mode_key: str) -> VectorStore:
        return build_store(provider, mode_key, ALL_TEXT)

    def get_qa_store(provider: str) -> Optional[VectorStore]:
        if not qa_items: return None
        return build_store(provider, "qa", {}, qa_items)

    def get_qa_hint(provider: str, query: str) -> Optional[str]:
        store = get_qa_store(provider)
        if not store: return None
        hits = store.search(query, k=1)
        if not hits: return None
        txt = hits[0]["text"].replace("[QA MEMORY]", "").strip()
        return txt

    # Runner per engine
    def run_engine(provider: str):
        store_std = get_store(provider, "standard")
        store_ctx = get_store(provider, "contextual")

        # Retrieval
        if mode == "Standard":
            pdf_hits = retrieve_standard(store_std, query, top_k)
        elif mode == "Contextual":
            pdf_hits = retrieve_contextual(store_ctx, query, top_k)
        elif mode == "Query Transformation":
            pdf_hits = retrieve_query_transform(provider, store_std, query, top_k)
        elif mode == "Adaptive":
            _, pdf_hits = retrieve_adaptive(provider, store_ctx, store_std, query, top_k)
        else:  # Combined
            pdf_hits = retrieve_combined(provider, store_std, store_ctx, query, top_k, enable_transforms)

        qa_hint = get_qa_hint(provider, query) if use_qa_memory else None
        ans = answer(provider, query, pdf_hits, qa_hint)

        # Scoring vs retrieved context
        pick_store = store_std if mode == "Standard" else (store_ctx if mode == "Contextual" else None)
        sim_centroid, sim_max = answer_similarity_against_hits(provider, ans, pick_store, pdf_hits)

        return ans, pdf_hits, sim_centroid, sim_max, ("Standard" if mode=="Standard" else "Contextual" if mode=="Contextual" else mode)

    # Run button
    if st.button("Run"):
        cols = st.columns(3) if "Both" in compare_choice else (st.columns(2) if "Compare" in compare_choice else st.columns(1))
        results = []

        if compare_choice == "OpenAI":
            results = [("OpenAI",) + run_engine("OpenAI")]
        elif compare_choice == "Gemini":
            results = [("Gemini",) + run_engine("Gemini")]
        elif compare_choice == "Compare: OpenAI vs Gemini":
            results = [
                ("OpenAI",) + run_engine("OpenAI"),
                ("Gemini",) + run_engine("Gemini"),
            ]
        else:
            # Both: merge contexts from both engines and synthesize a consensus
            ans_o, hits_o, simc_o, simm_o, tag_o = run_engine("OpenAI")
            ans_g, hits_g, simc_g, simm_g, tag_g = run_engine("Gemini")

            merged = {}
            for h in hits_o + hits_g:
                key = (h["source"], h["chunk_id"])
                if key not in merged or h["similarity"] > merged[key]["similarity"]:
                    merged[key] = h
            hits_both = sorted(merged.values(), key=lambda x: -x["similarity"])[:top_k]
            qa_hint_both = get_qa_hint("OpenAI" if OPENAI_API_KEY else "Gemini", query) if use_qa_memory else None
            ctx_engine = "OpenAI" if OPENAI_API_KEY else "Gemini"
            ans_both = answer(ctx_engine, query, hits_both, qa_hint_both)

            # For merged, re-embed hit texts with the synthesis engine for fair scoring
            simc_b, simm_b = answer_similarity_against_hits(ctx_engine, ans_both, None, hits_both)

            results = [
                ("OpenAI", ans_o, hits_o, simc_o, simm_o, tag_o),
                ("Gemini", ans_g, hits_g, simc_g, simm_g, tag_g),
                ("Both",   ans_both, hits_both, simc_b, simm_b, "Merged"),
            ]

        # Render
        for col, (name, ans, hits, simc, simm, tag) in zip(cols, results):
            with col:
                st.subheader(f"{name} — {tag}")
                st.markdown("**Answer**")
                st.write(ans if ans else "(no answer — check API limits/secrets)")
                st.markdown(f"**Avg similarity (top-{top_k}):** `{avg_sim(hits):.3f}`")
                st.markdown(f"**Ans↔Context similarity:** centroid=`{simc:.3f}` · max=`{simm:.3f}`")
                with st.expander("Show retrieved context"):
                    for i, h in enumerate(hits, 1):
                        st.markdown(f"**{i}. {os.path.basename(h['source'])} — chunk {h['chunk_id']} — sim {h['similarity']:.3f}**")
                        st.write(h["text"])
