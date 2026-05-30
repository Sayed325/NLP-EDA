"""
Reusable helpers for the RAG EDA notebook.
All loaders and stat-computers live here so notebook cells stay focused on analysis.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")

KEYS     = ["apple_2023", "apple_2024", "apple_2025",
            "google_2023", "google_2024", "google_2025"]
COMPANIES = ["apple", "google"]
YEARS     = [2023, 2024, 2025]

# Tableau-inspired palette kept consistent across all plots
YEAR_COLORS    = {2023: "#4e79a7", 2024: "#f28e2b", 2025: "#59a14f"}
COMPANY_COLORS = {"apple": "#4e79a7", "google": "#e15759"}


def load_all(data_dir: Path = DATA_DIR):
    """Load all 12 JSON files. Returns (docs, chunks) dicts keyed by 'company_year'."""
    docs, chunks = {}, {}
    for key in KEYS:
        with open(data_dir / f"{key}.json", encoding="utf-8") as f:
            docs[key] = json.load(f)
        with open(data_dir / f"{key}_chunks.json", encoding="utf-8") as f:
            raw = json.load(f)
            chunks[key] = raw["chunks"]
    return docs, chunks


def compute_corpus_stats(docs: dict, chunks: dict) -> pd.DataFrame:
    """
    Build the per-document summary DataFrame used in Steps 1 and 7.
    Token stats come from chunk metadata; content-block counts from parsed docs.
    """
    rows = []
    for key in KEYS:
        doc = docs[key]
        cks = chunks[key]
        tokens  = [c["token_estimate"] for c in cks]
        n       = len(cks)
        table_n = sum(1 for c in cks if c["hasTables"])
        rows.append({
            "document":            key,
            "section_count":       len(doc["sections"]),
            "content_block_count": sum(len(s["content_blocks"]) for s in doc["sections"]),
            "chunk_count":         n,
            "avg_token_estimate":  round(float(np.mean(tokens)), 1),
            "min_token":           int(np.min(tokens)),
            "max_token":           int(np.max(tokens)),
            "table_chunk_count":   table_n,
            "table_chunk_pct":     round(100 * table_n / n, 1),
            "tiny_chunk_count":    sum(1 for t in tokens if t < 200),
        })
    return pd.DataFrame(rows)


def save_fig(fig: plt.Figure, fname: str, dpi: int = 150) -> Path:
    """Save figure to output/ directory and print confirmation."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / fname
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved -> {path}")
    return path
