# app.py (fast, compare OpenAI vs Gemini vs Both, uses repo-root PDFs + vector pickles)
import os
import time
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st

# =========================
# Config — use ALL PDFs at repo root
# =========================
def list_repo_pdfs() -> List[str]:
    return sorted([f for f in os.listdir(".") if f.lower().endswith(".pdf")])

PDF_FILES = list_repo_pdfs()
CHUNK_SIZE = 900
OVERLAP = 250
TOP_K_DEFAULT = 4

# Vector pickle names at repo root
PREBUILT = {
    "OpenAI": "openai_vectors.pkl",
    "Gemini": "gemini_vectors.pkl",
}

# =========================
# Secrets (no widgets)
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    st.error("Both OPENAI_API_KEY and GEMINI_API_KEY are missing. Add at least one in secrets.")
    st.stop()

# =========================
# Lazy imports
# =========================
def _lazy_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def _lazy_genai():
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai

# =========================
# Utils
# =========================
def avg_sim(hits: List[dict]) -> float:
    if not hits: return 0.0
    return float(np.mean([h.get("similarity", 0.0) for h in hits]))

# =========================
# PDF -> text
# =========================
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

# =========================
# Embeddings (fast as possible)
# =========================
def embed_openai_many(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    OpenAI supports batching — big speedup.
    """
    client = _lazy_openai()
    model = "text-embedding-3-small"
    vecs = []
    # batch to reduce round-trips
    for i in range(0, len(texts), batch_size):
        chunk = [t[:3000] for t in texts[i:i+batch_size]]
        for _ in range(4):
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                for item in resp.data:
                    vecs.append(np.array(item.embedding, dtype=np.float32))
                break
            except Exception:
                time.sleep(2)
        else:
            # fallback zeros for the whole batch
            vec_len = 1536
            for _ in chunk:
                vecs.append(np.zeros((vec_len,), dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 1536), dtype=np.float32)

def embed_gemini_many(texts: List[str]) -> np.ndarray:
    """
    Gemini embedContent is per-text; keep retries small to be fast.
    """
    genai = _lazy_genai()
    model = "models/embedding-001"
    vecs = []
    for t in texts:
        t2 = t[:3000]
        for _ in range(3):
            try:
                r = genai.embed_content(model=model, content=t2)
                vec = r.get("embedding") if isinstance(r, dict) else getattr(r, "embedding", {}).get("values", [])
                if isinstance(vec, dict) and "values" in vec:
                    vec = vec["values"]
                vecs.append(np.array(vec, dtype=np.float32))
                break
            except Exception:
                time.sleep(1)
        else:
            vecs.append(np.zeros((768,), dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)

# =========================
# Vector store
# =========================
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

def build_corpus(all_text: Dict[str, str]) -> Tuple[List[str], List[dict]]:
    """
    Fast-Combined: we only build contextual chunks (higher recall without extra LLM calls).
    """
    texts, metas = [], []
    for fname, full in all_text.items():
        if not full: 
            continue
        base = chunk_text(full)
        ctx_chunks = contextualize_chunks(base, window=1)
        for i, ch in enumerate(ctx_chunks):
            # include filename header (helps retrieval slightly)
            wrapped = f"[SOURCE: {os.path.basename(fname)} | chunk {i}]\n{ch}"
            texts.append(wrapped)
            metas.append({"source": fname, "chunk_id": i, "text": wrapped})
    return texts, metas

@st.cache_resource(show_spinner=False)
def build_store(provider: str, all_text: Dict[str, str]) -> VectorStore:
    # 1) Load prebuilt if present
    pre = load_prebuilt(provider)
    if pre:
        return pre

    # 2) Build once (contextual only)
    texts, metas = build_corpus(all_text)
    if not texts:
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])

    if provider == "OpenAI":
        embs = embed_openai_many(texts)
    else:
        embs = embed_gemini_many(texts)
    store = VectorStore(provider, embs, metas)
    save_prebuilt(provider, store)
    return store

def rebuild_and_save(provider: str, all_text: Dict[str, str]) -> VectorStore:
    texts, metas = build_corpus(all_text)
    if not texts:
        return VectorStore(provider, np.zeros((0,1), dtype=np.float32), [])
    if provider == "OpenAI":
        embs = embed_openai_many(texts)
    else:
        embs = embed_gemini_many(texts)
    store = VectorStore(provider, embs, metas)
    save_prebuilt(provider, store)
    return store

# =========================
# LLM answerer (few calls)
# =========================
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

def answer(engine: str, query: str, hits: List[dict]) -> str:
    ctx = "\n\n---\n\n".join([h["text"] for h in hits]) if hits else "(no context)"
    prompt = (
        "Use ONLY the provided context to answer. If it doesn't contain enough info, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    return llm(engine, prompt, 0.1)

# =========================
# UI
# =========================
st.set_page_config(page_title="PA 211 / PEMA RAG — Fast Compare", layout="wide")
st.title("PA 211 / PEMA RAG — Fast Compare (OpenAI vs Gemini vs Both)")

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

compare_choice = st.radio(
    "Engine",
    ["OpenAI", "Gemini", "Compare: OpenAI vs Gemini", "Compare: OpenAI vs Gemini vs Both"],
    index=2
)
top_k = st.slider("Top-K passages", 1, 8, TOP_K_DEFAULT)

default_q = "What does the Red Cross and PEMA shelter guide say about bringing pets to emergency shelters in Harrisburg?"
query = st.text_area("Your question", value=default_q, height=80)

# Load text once
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
    with st.spinner("Rebuilding OpenAI vectors… (contextual only)"):
        _ = rebuild_and_save("OpenAI", ALL_TEXT)
        st.success("OpenAI vectors rebuilt and saved.")
if rebuild_gemini:
    with st.spinner("Rebuilding Gemini vectors… (contextual only)"):
        _ = rebuild_and_save("Gemini", ALL_TEXT)
        st.success("Gemini vectors rebuilt and saved.")

def get_store(provider: str) -> VectorStore:
    return build_store(provider, ALL_TEXT)

def run_engine(provider: str):
    store = get_store(provider)
    hits = store.search(query, k=top_k)
    ans = answer(provider, query, hits)
    return ans, hits, "Fast Combined"

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
        # Both: merge top contexts from both stores; synthesize with OpenAI if available else Gemini
        ans_o, hits_o, _ = run_engine("OpenAI")
        ans_g, hits_g, _ = run_engine("Gemini")
        merged = {}
        for h in hits_o + hits_g:
            key = (h["source"], h["chunk_id"])
            if key not in merged or h["similarity"] > merged[key]["similarity"]:
                merged[key] = h
        hits_both = sorted(merged.values(), key=lambda x: -x["similarity"])[:top_k]
        ctx = "\n\n---\n\n".join([h["text"] for h in hits_both]) if hits_both else "(no context)"
        synth_engine = "OpenAI" if OPENAI_API_KEY else "Gemini"
        synth_prompt = f"Combine the evidence from both contexts and answer concisely.\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        ans_both = llm(synth_engine, synth_prompt, 0.1)
        results = [
            ("OpenAI", ans_o, hits_o, "Fast Combined"),
            ("Gemini", ans_g, hits_g, "Fast Combined"),
            ("Both",   ans_both, hits_both, "Merged"),
        ]

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
