import os
import time
import json
import math
import pickle
import functools
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st

# -----------------------------
# Config — PDFs live at repo root
# -----------------------------
PDF_FILES = [
    "211 RESPONDS TO URGENT NEEDS.pdf",
    "PA 211 Disaster Community Resources.pdf",
    "PEMA.pdf",
    "ready-gov_disaster-preparedness-guide-for-older-adults.pdf",
    "Substantial Damages Toolkit.pdf",
]

# -----------------------------
# Secrets (no widgets)
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    st.stop()  # fail early if both missing (so no confusing errors later)

# -----------------------------
# Imports for LLMs (lazy)
# -----------------------------
def _lazy_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def _lazy_genai():
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# -----------------------------
# Utility: cosine similarity
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def rank_by_cosine(query_vec: np.ndarray, docs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sims = (docs @ query_vec) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query_vec) + 1e-9)
    order = np.argsort(-sims)
    return order, sims

# -----------------------------
# PDF → text
# -----------------------------
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(path)
    except Exception as e:
        return ""
    chunks = []
    for p in doc:
        try:
            chunks.append(p.get_text("text"))
        except Exception:
            continue
    doc.close()
    return "\n".join(chunks).strip()

@st.cache_data(show_spinner=False)
def load_all_pdfs_text(pdf_files: List[str]) -> Dict[str, str]:
    out = {}
    for p in pdf_files:
        if os.path.exists(p):
            out[p] = extract_text_from_pdf(p)
        else:
            out[p] = ""
    return out

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, n: int = 1000, overlap: int = 200) -> List[str]:
    if not text: return []
    chunks = []
    step = max(1, n - overlap)
    for i in range(0, len(text), step):
        ch = text[i:i + n]
        if ch.strip():
            chunks.append(ch)
    return chunks

def contextualize_chunks(chunks: List[str], window: int = 1) -> List[str]:
    if not chunks: return []
    out = []
    for i, ch in enumerate(chunks):
        ctx = []
        for w in range(i - window, i + window + 1):
            if 0 <= w < len(chunks):
                ctx.append(chunks[w])
        out.append("\n\n".join(ctx))
    return out

# -----------------------------
# Embeddings (OpenAI / Gemini)
# -----------------------------
EMBED_CACHE_DIR = ".rag_cache"
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

def _cache_path(model_name: str) -> str:
    safe = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(EMBED_CACHE_DIR, f"{safe}.pkl")

def _hash_corpus(texts: List[str]) -> str:
    # quick content hash (don’t overthink)
    return str(abs(hash("||".join([t[:1000] for t in texts]))))

def _load_cached_vectors(model_name: str, corpus_hash: str):
    path = _cache_path(model_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.get("corpus_hash") == corpus_hash:
            return obj["embeddings"], obj["metas"]
    except Exception:
        return None
    return None

def _save_cached_vectors(model_name: str, corpus_hash: str, embeddings: np.ndarray, metas: List[dict]):
    path = _cache_path(model_name)
    try:
        with open(path, "wb") as f:
            pickle.dump({"corpus_hash": corpus_hash, "embeddings": embeddings, "metas": metas}, f)
    except Exception:
        pass

def embed_openai(texts: List[str]) -> np.ndarray:
    client = _lazy_openai()
    vecs = []
    model = "text-embedding-3-small"
    for t in texts:
        t2 = t[:3000]
        for _ in range(3):  # simple retry
            try:
                resp = client.embeddings.create(model=model, input=t2)
                vecs.append(np.array(resp.data[0].embedding, dtype=np.float32))
                break
            except Exception as e:
                time.sleep(2)
        else:
            vecs.append(np.zeros((1536,), dtype=np.float32))
    return np.vstack(vecs)

def embed_gemini(texts: List[str]) -> np.ndarray:
    genai = _lazy_genai()
    model = "models/embedding-001"
    vecs = []
    for t in texts:
        t2 = t[:3000]
        for _ in range(3):
            try:
                r = genai.embed_content(model=model, content=t2)
                vecs.append(np.array(r["embedding"], dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
        else:
            # fallback vector
            vecs.append(np.zeros((768,), dtype=np.float32))
    return np.vstack(vecs)

# -----------------------------
# VectorStore abstraction
# -----------------------------
class VectorStore:
    def __init__(self, model_name: str, embeddings: np.ndarray, metas: List[dict]):
        self.model_name = model_name
        self.embeddings = embeddings
        self.metas = metas
        self.dim = embeddings.shape[1] if embeddings is not None and embeddings.size else 0

    def query_embed(self, text: str) -> np.ndarray:
        if "openai" in self.model_name.lower() or self.dim in (1536, 3072):  # crude check
            return embed_openai([text])[0]
        else:
            return embed_gemini([text])[0]

    def search(self, query: str, k: int = 4) -> List[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        q = self.query_embed(query)
        order, sims = rank_by_cosine(q, self.embeddings)
        out = []
        for idx in order[:k]:
            m = dict(self.metas[idx])
            m["similarity"] = float(sims[idx])
            out.append(m)
        return out

@st.cache_resource(show_spinner=False)
def build_store(engine: str, all_docs: Dict[str, str], mode: str) -> VectorStore:
    # mode in {"standard","contextual"}
    all_chunks = []
    metas = []
    for fname, text in all_docs.items():
        if not text: 
            continue
        base_chunks = chunk_text(text, n=1000, overlap=200)
        chunks = base_chunks if mode == "standard" else contextualize_chunks(base_chunks, window=1)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metas.append({"source": fname, "chunk_id": i, "text": ch})

    if not all_chunks:
        return VectorStore(f"{engine}-empty", np.zeros((0, 1), dtype=np.float32), [])

    model_key = "openai" if engine == "OpenAI" else "gemini"
    corpus_hash = _hash_corpus([m["text"] for m in metas])
    cached = _load_cached_vectors(model_key + "_" + mode, corpus_hash)
    if cached:
        emb, meta_cached = cached
        return VectorStore(model_key + "_" + mode, emb, meta_cached)

    if engine == "OpenAI":
        emb = embed_openai([m["text"] for m in metas])
    else:
        emb = embed_gemini([m["text"] for m in metas])

    _save_cached_vectors(model_key + "_" + mode, corpus_hash, emb, metas)
    return VectorStore(model_key + "_" + mode, emb, metas)

# -----------------------------
# Query transforms
# -----------------------------
def llm_generate(engine: str, prompt: str, temperature: float = 0.2) -> str:
    if engine == "OpenAI":
        client = _lazy_openai()
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return r.choices[0].message.content.strip()
            except Exception:
                time.sleep(2)
        return ""
    else:
        genai = _lazy_genai()
        model = genai.GenerativeModel("gemini-2.0-flash")
        for _ in range(3):
            try:
                r = model.generate_content(prompt)
                return (r.text or "").strip()
            except Exception:
                time.sleep(2)
        return ""

def transform_query(engine: str, query: str, ttype: str) -> List[str]:
    if ttype == "rewrite":
        p = f"Rewrite the following query to be clearer and more specific, but keep intent.\nQuery: {query}\nRewritten:"
        return [llm_generate(engine, p)]
    if ttype == "step_back":
        p = f"Create a high-level step-back version of this query so retrieval covers broader facts.\nQuery: {query}\nStep-back:"
        return [llm_generate(engine, p)]
    if ttype == "decompose":
        p = f"Decompose the query into 3 short sub-questions, one per line.\nQuery: {query}\nSub-questions:"
        ans = llm_generate(engine, p)
        lines = [l.strip("-• ").strip() for l in ans.splitlines() if l.strip()]
        return lines[:3] if lines else [query]
    return [query]

# -----------------------------
# Adaptive retrieval (lightweight rules to avoid extra LLM calls)
# -----------------------------
def classify_query_rule(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["why", "how", "compare", "tradeoff", "pros", "cons"]):
        return "Analytical"
    if any(w in q for w in ["i think", "opinion", "should i", "what do you think"]):
        return "Opinion"
    if any(w in q for w in ["me", "my", "our", "we", "in 17104", "near me"]):
        return "Contextual"
    return "Factual"

def adaptive_retrieve(engine: str, store: VectorStore, query: str, k: int = 4) -> Tuple[str, List[dict]]:
    qtype = classify_query_rule(query)
    queries = [query]
    if qtype == "Factual":
        queries = transform_query(engine, query, "rewrite")
    elif qtype == "Analytical":
        queries = transform_query(engine, query, "decompose")
    elif qtype == "Opinion":
        # pull diverse angles: pretend with step_back + original
        queries = [query] + transform_query(engine, query, "step_back")
    elif qtype == "Contextual":
        # add mini-contextualization
        queries = [f"{query} (community resources, service eligibility, shelter rules)"]
    hits = []
    seen = set()
    for q in queries:
        for h in store.search(q, k=k):
            key = (h["source"], h["chunk_id"])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    # sort by similarity and cap
    hits = sorted(hits, key=lambda x: -x["similarity"])[:k]
    return qtype, hits

# -----------------------------
# Answer generation
# -----------------------------
def answer_with_engine(engine: str, query: str, contexts: List[dict]) -> str:
    ctx = "\n\n---\n\n".join([c["text"] for c in contexts]) if contexts else "(no context found)"
    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context lacks the answer, say you don't have enough info.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    return llm_generate(engine, prompt, temperature=0.1)

# -----------------------------
# Scoring (similarity summary)
# -----------------------------
def summarize_similarity(contexts: List[dict]) -> float:
    if not contexts: return 0.0
    sims = [c.get("similarity", 0.0) for c in contexts]
    return float(np.mean(sims))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PA 211 / PEMA RAG (OpenAI vs Gemini)", layout="wide")
st.title("PA 211 / PEMA RAG — OpenAI vs Gemini vs Both")

with st.expander("PDF status", expanded=False):
    missing = [p for p in PDF_FILES if not os.path.exists(p)]
    if missing:
        st.error("Missing files:\n" + "\n".join(missing))
    else:
        st.success("All expected PDFs are present.")
    st.write("Using these PDFs:")
    st.write(PDF_FILES)

# Preload text (cached)
with st.spinner("Loading PDFs..."):
    all_text = load_all_pdfs_text(PDF_FILES)

# Retrieval mode
mode = st.selectbox(
    "Retrieval mode",
    ["Standard", "Contextual", "Query Transformation", "Adaptive", "Combined"],
    index=0
)

# Engines to compare
compare_choice = st.radio(
    "Engine",
    ["OpenAI", "Gemini", "Compare: OpenAI vs Gemini", "Compare: OpenAI vs Gemini vs Both"],
    index=2
)

top_k = st.slider("Top-K passages", 1, 8, 4)

default_q = "What does the Red Cross and PEMA shelter guide say about bringing pets to emergency shelters in Harrisburg?"
query = st.text_area("Your question", value=default_q, height=80)

# Build stores (cached)
def get_store(engine: str, mode: str) -> VectorStore:
    store_mode = "standard"
    if mode == "Contextual":
        store_mode = "contextual"
    elif mode == "Combined":
        # Build contextual store; (combined uses both below anyway)
        store_mode = "contextual"
    return build_store(engine, all_text, store_mode)

if st.button("Run"):
    cols = st.columns(3) if "Both" in compare_choice else st.columns(2) if "Compare" in compare_choice else st.columns(1)

    # Helper to run a single engine
    def run_engine(engine_name: str):
        # pick store for mode
        if mode == "Combined":
            store_std = build_store(engine_name, all_text, "standard")
            store_ctx = build_store(engine_name, all_text, "contextual")
            # query transforms
            transformed = transform_query(engine_name, query, "rewrite") + transform_query(engine_name, query, "step_back")
            # gather hits
            seen = set()
            hits = []
            # standard
            for h in store_std.search(query, k=top_k):
                key = (h["source"], h["chunk_id"]); 
                if key not in seen: seen.add(key); hits.append(h)
            # contextual
            for h in store_ctx.search(query, k=top_k):
                key = (h["source"], h["chunk_id"]); 
                if key not in seen: seen.add(key); hits.append(h)
            # transformed
            for tq in transformed:
                for h in store_std.search(tq, k=max(2, top_k//2)):
                    key = (h["source"], h["chunk_id"]); 
                    if key not in seen: seen.add(key); hits.append(h)
            # sort and trim
            hits = sorted(hits, key=lambda x: -x["similarity"])[:top_k]
            ans = answer_with_engine(engine_name, query, hits)
            score = summarize_similarity(hits)
            return ans, hits, score, "Combined"
        elif mode == "Adaptive":
            store = get_store(engine_name, "Standard")
            qtype, hits = adaptive_retrieve(engine_name, store, query, k=top_k)
            ans = answer_with_engine(engine_name, query, hits)
            score = summarize_similarity(hits)
            return ans, hits, score, f"Adaptive ({qtype})"
        elif mode == "Query Transformation":
            store = get_store(engine_name, "Standard")
            # try all three transforms and merge top hits
            tqs = (transform_query(engine_name, query, "rewrite")
                   + transform_query(engine_name, query, "step_back")
                   + transform_query(engine_name, query, "decompose"))
            seen = set(); hits = []
            for tq in tqs:
                for h in store.search(tq, k=max(2, top_k//2+1)):
                    key = (h["source"], h["chunk_id"])
                    if key not in seen:
                        seen.add(key); hits.append(h)
            hits = sorted(hits, key=lambda x: -x["similarity"])[:top_k]
            ans = answer_with_engine(engine_name, query, hits)
            score = summarize_similarity(hits)
            return ans, hits, score, "Query Transformation"
        else:
            # Standard or Contextual
            store = get_store(engine_name, mode)
            hits = store.search(query, k=top_k)
            ans = answer_with_engine(engine_name, query, hits)
            score = summarize_similarity(hits)
            return ans, hits, score, mode

    # Run based on choice
    results = []

    if compare_choice == "OpenAI":
        results = [("OpenAI",) + run_engine("OpenAI")]
    elif compare_choice == "Gemini":
        results = [("Gemini",) + run_engine("Gemini")]
    elif compare_choice == "Compare: OpenAI vs Gemini":
        results = [("OpenAI",) + run_engine("OpenAI"), ("Gemini",) + run_engine("Gemini")]
    else:
        # Both = blend contexts from OpenAI + Gemini embeddings into unified "Both"
        # simple union: get hits from both, then answer with OpenAI (or Gemini) as "Both"
        ans_o, hits_o, score_o, label_o = run_engine("OpenAI")
        ans_g, hits_g, score_g, label_g = run_engine("Gemini")
        # merge hits by source/chunk, keeping max similarity
        merged = {}
        for h in hits_o + hits_g:
            key = (h["source"], h["chunk_id"])
            if key not in merged or h["similarity"] > merged[key]["similarity"]:
                merged[key] = h
        hits_both = sorted(merged.values(), key=lambda x: -x["similarity"])[:top_k]
        # Answer with OpenAI for "Both" (choice is arbitrary but stable)
        ans_both = answer_with_engine("OpenAI", query, hits_both)
        score_both = summarize_similarity(hits_both)
        results = [
            ("OpenAI", ans_o, hits_o, score_o, label_o),
            ("Gemini", ans_g, hits_g, score_g, label_g),
            ("Both",   ans_both, hits_both, score_both, "Merged")
        ]

    # Render columns
    for (col, item) in zip(cols, results):
        engine_name, ans, hits, score, tag = item
        with col:
            st.subheader(f"{engine_name} — {tag}")
            st.markdown("**Answer**")
            st.write(ans if ans else "(no answer — check your API limits/secrets)")
            st.markdown(f"**Avg similarity of top-{top_k}:** `{score:.3f}`")
            with st.expander("Show retrieved context"):
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**{i}. {h['source']} — chunk {h['chunk_id']} — sim {h['similarity']:.3f}**")
                    st.write(h["text"])
