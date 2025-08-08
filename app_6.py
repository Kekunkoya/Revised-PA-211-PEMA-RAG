# app.py
import os
import time
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st

# =============== Config ===============
# We use ALL PDFs in the repo root:
def list_repo_pdfs() -> List[str]:
    return sorted([f for f in os.listdir(".") if f.lower().endswith(".pdf")])

PDF_FILES = list_repo_pdfs()

# Vector pickle names at repo root:
PREBUILT = {
    "OpenAI": "openai_vectors.pkl",
    "Gemini": "gemini_vectors.pkl",
}

# Chunking defaults (tuned a bit for recall)
CHUNK_SIZE = 900
OVERLAP = 250

# =============== Secrets (no widgets) ===============
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    st.error("Both OPENAI_API_KEY and GEMINI_API_KEY are missing. Add at least one in secrets.")
    st.stop()

# =============== Lazy imports ===============
def _lazy_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def _lazy_genai():
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# =============== Utils ===============
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def avg_sim(hits: List[dict]) -> float:
    if not hits: return 0.0
    return float(np.mean([h.get("similarity", 0.0) for h in hits]))

# =============== PDF -> Text ===============
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

# =============== Embeddings ===============
def embed_openai_many(texts: List[str]) -> np.ndarray:
    client = _lazy_openai()
    model = "text-embedding-3-small"
    embs = []
    for t in texts:
        t2 = t[:3000]
        for _ in range(4):
            try:
                r = client.embeddings.create(model=model, input=[t2])
                embs.append(np.array(r.data[0].embedding, dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
        else:
            embs.append(np.zeros((1536,), dtype=np.float32))
    return np.vstack(embs) if embs else np.zeros((0, 1536), dtype=np.float32)

def embed_gemini_many(texts: List[str]) -> np.ndarray:
    genai = _lazy_genai()
    model = "models/embedding-001"
    embs = []
    for t in texts:
        t2 = t[:3000]
        for _ in range(4):
            try:
                r = genai.embed_content(model=model, content=t2)
                vec = r.get("embedding") if isinstance(r, dict) else getattr(r, "embedding", {}).get("values", [])
                if isinstance(vec, dict) and "values" in vec:
                    vec = vec["values"]
                embs.append(np.array(vec, dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
        else:
            embs.append(np.zeros((768,), dtype=np.float32))
    return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)

# =============== Vector Store ===============
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
            out.append(m)
        return out

def prebuilt_path(provider: str) -> str:
    return PREBUILT.get(provider, "")

def load_prebuilt(provider: str) -> Optional[VectorStore]:
    p = prebuilt_path(provider)
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

def save_prebuilt(provider: str, store: VectorStore):
    p = prebuilt_path(provider)
    if not p:
        return
    try:
        with open(p, "wb") as f:
            pickle.dump({"embeddings": store.embeddings, "metas": store.metas}, f)
    except Exception:
        pass

def build_corpus(all_text: Dict[str, str], contextual=False) -> Tuple[List[str], List[dict]]:
    texts, metas = [], []
    for fname, full in all_text.items():
        if not full: 
            continue
        chunks = chunk_text(full)
        if contextual:
            chunks = contextualize_chunks(chunks, window=1)
        for i, ch in enumerate(chunks):
            wrapped = f"[SOURCE: {os.path.basename(fname)} | chunk {i}]\n{ch}"
            texts.append(wrapped)
            metas.append({"source": fname, "chunk_id": i, "text": wrapped})
    return texts, metas

@st.cache_resource(show_spinner=False)
def build_store(provider: str, all_text: Dict[str, str], contextual: bool) -> VectorStore:
    # 1) Try loading prebuilt
    pre = load_prebuilt(provider)
    if pre:
        return pre

    # 2) Build from PDFs
    texts, metas = build_corpus(all_text, contextual=contextual)
    if not texts: 
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])

    if provider == "OpenAI":
        embs = embed_openai_many(texts)
    else:
        embs = embed_gemini_many(texts)

    store = VectorStore(provider, embs, metas)
    save_prebuilt(provider, store)  # save to repo root for reuse
    return store

def rebuild_and_save(provider: str, all_text: Dict[str, str], contextual: bool) -> VectorStore:
    texts, metas = build_corpus(all_text, contextual=contextual)
    if not texts: 
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])
    if provider == "OpenAI":
        embs = embed_openai_many(texts)
    else:
        embs = embed_gemini_many(texts)
    store = VectorStore(provider, embs, metas)
    save_prebuilt(provider, store)
    return store

# =============== LLM helpers ===============
def llm(engine: str, prompt: str, temperature: float = 0.1) -> str:
    if engine == "OpenAI":
        client = _lazy_openai()
        for _ in range(4):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Answer only with the provided context."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception:
                time.sleep(2)
        return "ERROR"
    else:
        genai = _lazy_genai()
        mdl = genai.GenerativeModel("gemini-2.0-flash")
        for _ in range(4):
            try:
                r = mdl.generate_content(prompt)
                return (getattr(r, "text", "") or "").strip()
            except Exception:
                time.sleep(2)
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
        queries = [query] + transform_query(engine, query, "step_back")
    elif qtype == "Contextual":
        queries = [f"{query} (shelter rules, local eligibility, resources)"]
    seen = set(); hits = []
    for q in queries:
        for h in store.search(q, k=k):
            key = (h["source"], h["chunk_id"])
            if key not in seen:
                seen.add(key); hits.append(h)
    hits = sorted(hits, key=lambda x: -x["similarity"])[:k]
    return qtype, hits

def answer(engine: str, query: str, hits: List[dict]) -> str:
    ctx = "\n\n---\n\n".join([h["text"] for h in hits]) if hits else "(no context)"
    prompt = (
        "Answer using ONLY the provided context. If the context lacks the answer, say you don't have enough info.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    return llm(engine, prompt, 0.1)

# =============== UI ===============
st.set_page_config(page_title="PA 211 / PEMA RAG (PDFs in repo root)", layout="wide")
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
compare_choice = st.radio(
    "Engine",
    ["OpenAI", "Gemini", "Compare: OpenAI vs Gemini", "Compare: OpenAI vs Gemini vs Both"],
    index=2
)
top_k = st.slider("Top-K passages", 1, 8, 4)

default_q = "What does the Red Cross and PEMA shelter guide say about bringing pets to emergency shelters in Harrisburg?"
query = st.text_area("Your question", value=default_q, height=80)

# Load all PDF text
with st.spinner("Loading PDFs..."):
    ALL_TEXT = load_all_pdfs_text(PDF_FILES)

colb = st.columns(3)
with colb[0]:
    rebuild_openai = st.button("Build/Refresh OpenAI vectors")
with colb[1]:
    rebuild_gemini = st.button("Build/Refresh Gemini vectors")
with colb[2]:
    show_debug = st.checkbox("Show debug")

if rebuild_openai:
    with st.spinner("Rebuilding OpenAI vectors…"):
        _ = rebuild_and_save("OpenAI", ALL_TEXT, contextual=(mode in ["Contextual", "Combined"]))
        st.success("OpenAI vectors rebuilt and saved (repo root).")
if rebuild_gemini:
    with st.spinner("Rebuilding Gemini vectors…"):
        _ = rebuild_and_save("Gemini", ALL_TEXT, contextual=(mode in ["Contextual", "Combined"]))
        st.success("Gemini vectors rebuilt and saved (repo root).")

def get_store(provider: str, contextual: bool) -> VectorStore:
    return build_store(provider, ALL_TEXT, contextual=contextual)

def run_engine(provider: str):
    contextual = (mode in ["Contextual", "Combined"])

    if mode == "Combined":
        # Merge: standard + contextual + transforms
        store_std = get_store(provider, contextual=False)
        store_ctx = get_store(provider, contextual=True)
        hits = store_std.search(query, k=top_k)

        existing = {(h["source"], h["chunk_id"]) for h in hits}
        for h in store_ctx.search(query, k=top_k):
            key = (h["source"], h["chunk_id"])
            if key not in existing:
                hits.append(h); existing.add(key)

        # transforms
        tqs = transform_query(provider, query, "rewrite") + transform_query(provider, query, "step_back")
        for tq in tqs:
            for h in store_std.search(tq, k=max(2, top_k // 2)):
                key = (h["source"], h["chunk_id"])
                if key not in existing:
                    hits.append(h); existing.add(key)

        hits = sorted(hits, key=lambda x: -x["similarity"])[:top_k]
        ans = answer(provider, query, hits)
        return ans, hits, "Combined"

    if mode == "Adaptive":
        # Use contextual store for a bit more recall
        store = get_store(provider, contextual=True)
        qtype, hits = adaptive_retrieve(provider, store, query, k=top_k)
        ans = answer(provider, query, hits)
        return f"{ans}", hits, f"Adaptive ({qtype})"

    if mode == "Query Transformation":
        # use standard store, then merge transforms
        store_std = get_store(provider, contextual=False)
        tqs = (transform_query(provider, query, "rewrite") +
               transform_query(provider, query, "step_back") +
               transform_query(provider, query, "decompose"))
        seen = set(); hits = []
        for tq in tqs:
            for h in store_std.search(tq, k=max(2, top_k // 2 + 1)):
                key = (h["source"], h["chunk_id"])
                if key not in seen:
                    seen.add(key); hits.append(h)
        hits = sorted(hits, key=lambda x: -x["similarity"])[:top_k]
        ans = answer(provider, query, hits)
        return ans, hits, "Query Transformation"

    # Standard or Contextual only
    store = get_store(provider, contextual=(mode == "Contextual"))
    hits = store.search(query, k=top_k)
    ans = answer(provider, query, hits)
    return ans, hits, mode

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
        # Both: run each, then merge contexts and synthesize a consensus
        ans_o, hits_o, tag_o = run_engine("OpenAI")
        ans_g, hits_g, tag_g = run_engine("Gemini")
        merged = {}
        for h in hits_o + hits_g:
            key = (h["source"], h["chunk_id"])
            if key not in merged or h["similarity"] > merged[key]["similarity"]:
                merged[key] = h
        hits_both = sorted(merged.values(), key=lambda x: -x["similarity"])[:top_k]
        ctx = "\n\n---\n\n".join([h["text"] for h in hits_both]) if hits_both else "(no context)"
        synth_prompt = f"Combine the evidence from both contexts and answer concisely.\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans_both = llm("OpenAI" if OPENAI_API_KEY else "Gemini", synth_prompt, 0.1)
        results = [
            ("OpenAI", ans_o, hits_o, tag_o),
            ("Gemini", ans_g, hits_g, tag_g),
            ("Both",   ans_both, hits_both, "Merged"),
        ]

    # Render
    for col, (name, ans, hits, tag) in zip(cols, results):
        with col:
            st.subheader(f"{name} — {tag}")
            st.markdown("**Answer**")
            st.write(ans if ans else "(no answer — check API limits/secrets)")
            st.markdown(f"**Avg similarity (top-{top_k}):** `{avg_sim(hits):.3f}`")
            with st.expander("Show retrieved context"):
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**{i}. {os.path.basename(h['source'])} — chunk {h['chunk_id']} — sim {h['similarity']:.3f}**")
                    st.write(h["text"])
