#!/usr/bin/env python3
"""
Evaluate two system pairs (G12 vs 12, G4 vs 4) with BERTScore and cosine similarity.
Saves heatmaps and prints per-pair summary stats.

Usage:
  python evaluate_pairs.py

Edit the DATA INPUT SECTION to point at your files or paste lists in code.
"""

import os
import json
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score

# -------------------- Config --------------------
BERTSCORE_MODEL = "roberta-large"              # fallback: "roberta-base" if you hit OOM
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"

plt.rcParams["figure.figsize"] = (8, 6)
sns.set_context("talk")


# -------------------- Loaders --------------------
def read_txt_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def read_jsonl_list(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                items.append(obj)
            elif isinstance(obj, dict) and "text" in obj:
                items.append(str(obj["text"]))
            else:
                items.append(str(obj))
    return items

def read_csv_cols(path: str, col_a: str, col_b: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path)
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(f"Missing columns. Available: {df.columns.tolist()}")
    return df[col_a].astype(str).tolist(), df[col_b].astype(str).tolist()

def ensure_same_length(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


# -------------------- Scoring --------------------
def bertscore_matrix(cands: List[str], refs: List[str], model_type: str) -> np.ndarray:
    """
    Build |cands| x |refs| matrix of BERTScore F1 by scoring each candidate vs all refs.
    """
    m, n = len(cands), len(refs)
    out = np.zeros((m, n), dtype=np.float32)
    for i, cand in enumerate(cands):
        C = [cand] * n
        _, _, F1 = bert_score(C, refs, lang="en", model_type=model_type, verbose=False)
        out[i, :] = F1.numpy()
    return out

def cosine_matrix(a: List[str], b: List[str], embed_model: str) -> np.ndarray:
    model = SentenceTransformer(embed_model)
    A = model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    B = model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(A, B).cpu().numpy().astype(np.float32)
    return sim

def aligned_scores(mat: np.ndarray) -> np.ndarray:
    k = min(mat.shape[0], mat.shape[1])
    return np.array([mat[i, i] for i in range(k)], dtype=np.float32)


# -------------------- Plotting --------------------
def plot_heatmap(mat: np.ndarray, title: str, save_path: str,
                 x_labels: List[str] | None = None,
                 y_labels: List[str] | None = None):
    plt.figure(figsize=(max(6, (mat.shape[1] * 0.5)), max(5, (mat.shape[0] * 0.5))))
    ax = sns.heatmap(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("References / System B")
    ax.set_ylabel("Candidates / System A")
    if x_labels is not None and len(x_labels) <= 30:
        ax.set_xticklabels([str(i) for i in range(len(x_labels))], rotation=90)
    if y_labels is not None and len(y_labels) <= 30:
        ax.set_yticklabels([str(i) for i in range(len(y_labels))], rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# -------------------- Evaluation Driver --------------------
def evaluate_pair(A: List[str], B: List[str], tag: str):
    A, B = ensure_same_length(A, B)
    print(f"[{tag}] evaluating {len(A)} pairs")

    print(f"[{tag}] computing BERTScore matrix…")
    bert_mat = bertscore_matrix(A, B, model_type=BERTSCORE_MODEL)
    print(f"[{tag}] computing cosine similarity matrix…")
    cos_mat = cosine_matrix(A, B, embed_model=EMBED_MODEL)

    # Save heatmaps
    plot_heatmap(bert_mat, f"BERTScore F1 — {tag}", f"bert_heatmap_{tag}.png", B, A)
    plot_heatmap(cos_mat,  f"Cosine Similarity — {tag}", f"cosine_heatmap_{tag}.png", B, A)

    # Print aligned (diagonal) stats
    bert_diag = aligned_scores(bert_mat)
    cos_diag  = aligned_scores(cos_mat)
    print(f"[{tag}] BERTScore diag: mean={bert_diag.mean():.4f}  min={bert_diag.min():.4f}  max={bert_diag.max():.4f}")
    print(f"[{tag}] Cosine    diag: mean={cos_diag.mean():.4f}  min={cos_diag.min():.4f}  max={cos_diag.max():.4f}")

    # Save matrices for downstream analysis
    np.save(f"bert_matrix_{tag}.npy", bert_mat)
    np.save(f"cos_matrix_{tag}.npy", cos_mat)


# -------------------- DATA INPUT SECTION --------------------
def load_data_sources():
    """
    Replace these with your actual data sources.
    You can load from files or paste Python lists inline.
    """

    # Option A: paste lists directly
    G12 = [
        "Example prediction 1 from G12.",
        "Example prediction 2 from G12.",
        "Example prediction 3 from G12.",
    ]
    S12 = [
        "Example reference/system 12 text 1.",
        "Example reference/system 12 text 2.",
        "Example reference/system 12 text 3.",
    ]

    G4 = [
        "Example prediction 1 from G4.",
        "Example prediction 2 from G4.",
        "Example prediction 3 from G4.",
    ]
    S4 = [
        "Example reference/system 4 text 1.",
        "Example reference/system 4 text 2.",
        "Example reference/system 4 text 3.",
    ]

    # Option B: file-based (uncomment and point to your files)
    # G12 = read_txt_lines("G12.txt")
    # S12 = read_txt_lines("12.txt")
    # G4  = read_txt_lines("G4.txt")
    # S4  = read_txt_lines("4.txt")

    # Option C: JSONL or CSV examples:
    # G12 = read_jsonl_list("G12.jsonl")
    # S12 = read_jsonl_list("12.jsonl")
    # G4  = read_jsonl_list("G4.jsonl")
    # S4  = read_jsonl_list("4.jsonl")

    # preds_12, refs_12 = read_csv_cols("g12_12.csv", col_a="pred", col_b="ref")
    # preds_4,  refs_4  = read_csv_cols("g4_4.csv",  col_a="pred", col_b="ref")
    # G12, S12 = preds_12, refs_12
    # G4,  S4  = preds_4,  refs_4

    return (G12, S12, G4, S4)


# -------------------- Main --------------------
def main():
    # (Optional) argparse if you want to pass file paths via CLI later
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    G12, S12, G4, S4 = load_data_sources()

    evaluate_pair(G12, S12, tag="G12_vs_12")
    evaluate_pair(G4,  S4,  tag="G4_vs_4")

    print("\n✅ Done. Saved:")
    print("  - bert_heatmap_G12_vs_12.png, cosine_heatmap_G12_vs_12.png")
    print("  - bert_heatmap_G4_vs_4.png,  cosine_heatmap_G4_vs_4.png")
    print("  - bert_matrix_*.npy, cos_matrix_*.npy")

if __name__ == "__main__":
    main()
