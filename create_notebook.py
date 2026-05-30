"""
One-shot script: generates eda.ipynb from source strings.
Run once; re-run any time the cell content changes.
"""
import json, uuid

def _id():
    return uuid.uuid4().hex[:8]

def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }

# ─── cells ────────────────────────────────────────────────────────────────────
cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "# RAG EDA — Apple & Google 10-K Filings (2023–2025)\n\n"
    "Exploratory analysis of a financial RAG corpus: 6 annual 10-K filings\n"
    "(3 years × 2 companies) parsed into structured JSON and chunked for retrieval.\n"
    "The notebook measures chunk quality, section coverage, content characteristics,\n"
    "and keyword trends to inform downstream retrieval tuning."
))

# ── Setup / Load ───────────────────────────────────────────────────────────────
cells.append(md(
    "## Setup — Imports & Data Load\n\n"
    "Load all 12 JSON files into two dictionaries keyed by `company_year`.\n"
    "All downstream cells consume `docs` (parsed 10-K structure) and `chunks` (chunker output)."
))

cells.append(code(
    "import os\n"
    "import sys\n"
    "import re\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patches as mpatches\n"
    "import seaborn as sns\n"
    "from collections import Counter\n"
    "from pathlib import Path\n"
    "\n"
    "%matplotlib inline\n"
    "\n"
    "sys.path.insert(0, \".\")\n"
    "from eda_utils import (\n"
    "    DATA_DIR, OUTPUT_DIR, KEYS, COMPANIES, YEARS,\n"
    "    YEAR_COLORS, COMPANY_COLORS,\n"
    "    load_all, compute_corpus_stats, save_fig,\n"
    ")\n"
    "\n"
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
    "plt.rcParams.update({\"figure.dpi\": 120, \"font.size\": 10})\n"
    "sns.set_theme(style=\"whitegrid\", font_scale=1.0)\n"
    "\n"
    "docs, chunks = load_all()\n"
    "total_chunks = sum(len(v) for v in chunks.values())\n"
    "print(f\"Loaded {len(docs)} documents — {total_chunks} total chunks\")"
))

# ── Step 1 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 1 — Corpus Overview Table\n\n"
    "Baseline statistics for each document: section count, content block count, chunk count,\n"
    "token distribution summary, table coverage, and tiny-chunk count.\n\n"
    "High `tiny_chunk_count` relative to `chunk_count` signals passages the chunker couldn't\n"
    "fill to the minimum threshold — often short boilerplate sections — which are retrieval\n"
    "dead weight and worth filtering or merging in the pipeline."
))

cells.append(code(
    "df = compute_corpus_stats(docs, chunks)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(15, 3.5))\n"
    "ax.axis(\"off\")\n"
    "tbl = ax.table(\n"
    "    cellText=df.values.tolist(),\n"
    "    colLabels=df.columns.tolist(),\n"
    "    cellLoc=\"center\",\n"
    "    loc=\"center\",\n"
    ")\n"
    "tbl.auto_set_font_size(False)\n"
    "tbl.set_fontsize(8)\n"
    "tbl.scale(1, 2.0)\n"
    "for j in range(len(df.columns)):\n"
    "    tbl[0, j].set_facecolor(\"#2c3e50\")\n"
    "    tbl[0, j].set_text_props(color=\"white\", fontweight=\"bold\")\n"
    "ax.set_title(\"Corpus Overview — All Documents\", fontsize=13, pad=10)\n"
    "save_fig(fig, \"01_corpus_overview.png\")\n"
    "plt.show()\n"
    "plt.close(fig)\n"
    "df"
))

# ── Step 2 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 2 — Token Size Distribution\n\n"
    "A RAG system works best when chunks cluster near the target context window (here 800 tokens).\n"
    "Chunks below the minimum threshold (200) are too thin to carry meaningful context;\n"
    "chunks above the target risk overflow during retrieval re-ranking.\n\n"
    "Legend entries show the proportion of chunks below 200 tokens per document — a quick\n"
    "signal for years where the chunker produced unusually many micro-chunks."
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "for ax, company in zip(axes, COMPANIES):\n"
    "    for year in YEARS:\n"
    "        key = f\"{company}_{year}\"\n"
    "        tokens = [c[\"token_estimate\"] for c in chunks[key]]\n"
    "        tiny_pct = 100 * sum(t < 200 for t in tokens) / len(tokens)\n"
    "        ax.hist(tokens, bins=30, alpha=0.6,\n"
    "                label=f\"{year}  (tiny: {tiny_pct:.0f}%)\",\n"
    "                color=YEAR_COLORS[year])\n"
    "    ax.axvline(200, color=\"red\",   linestyle=\"--\", linewidth=1.4,\n"
    "               label=\"min threshold (200)\")\n"
    "    ax.axvline(800, color=\"green\", linestyle=\"--\", linewidth=1.4,\n"
    "               label=\"target window (800)\")\n"
    "    ax.set_title(f\"{company.title()} — Token Distribution\")\n"
    "    ax.set_xlabel(\"Token Estimate\")\n"
    "    ax.set_ylabel(\"Chunk Count\")\n"
    "    ax.legend(fontsize=8)\n"
    "\n"
    "fig.suptitle(\"Token Size Distribution by Company and Year\", fontsize=13)\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"02_token_distribution.png\")\n"
    "plt.show()\n"
    "plt.close(fig)"
))

# ── Step 3 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 3 — Section Coverage Heatmap\n\n"
    "Each 10-K is divided into standardised items (Business, Risk Factors, MD&A,\n"
    "Financial Statements, etc.). This heatmap shows chunk count per item per document.\n\n"
    "Dark cells = well-covered sections (good for RAG). White/light cells = items with\n"
    "zero chunks — retrieval blind spots that will produce irrelevant results if a user\n"
    "asks a question anchored to those sections."
))

cells.append(code(
    "def _item_key(x):\n"
    "    m = re.match(r'Item\\s+(\\d+)(.*)', x)\n"
    "    return (int(m.group(1)), m.group(2).strip()) if m else (999, x)\n"
    "\n"
    "all_items = sorted(\n"
    "    set(c[\"item\"] for key in KEYS for c in chunks[key] if c.get(\"item\")),\n"
    "    key=_item_key,\n"
    ")\n"
    "\n"
    "matrix = pd.DataFrame(\n"
    "    {key: Counter(c[\"item\"] for c in chunks[key] if c.get(\"item\")) for key in KEYS},\n"
    "    index=all_items,\n"
    ").fillna(0).astype(int)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(12, 8))\n"
    "sns.heatmap(\n"
    "    matrix, annot=True, fmt=\"d\", cmap=\"YlOrRd\",\n"
    "    linewidths=0.4, linecolor=\"#cccccc\",\n"
    "    cbar_kws={\"label\": \"Chunk Count\"},\n"
    "    ax=ax,\n"
    ")\n"
    "ax.set_title(\"Chunk Count by Section \\u00d7 Document\", fontsize=14)\n"
    "ax.set_xlabel(\"Document\")\n"
    "ax.set_ylabel(\"10-K Item\")\n"
    "ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha=\"right\")\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"03_section_heatmap.png\")\n"
    "plt.show()\n"
    "plt.close(fig)"
))

# ── Step 4 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 4 — Content Type Breakdown\n\n"
    "The chunker labels each chunk as `prose` (paragraphs only) or `mixed`\n"
    "(paragraphs + embedded tables). Mixed chunks demand different embedding strategies:\n"
    "table data benefits from structured or hybrid lookup rather than pure dense retrieval.\n\n"
    "The lower panel shows total tables referenced across all chunks per document —\n"
    "a rough proxy for how quantitative each filing is."
))

cells.append(code(
    "prose_counts = [sum(1 for c in chunks[k] if c[\"content_type\"] == \"prose\") for k in KEYS]\n"
    "mixed_counts = [sum(1 for c in chunks[k] if c[\"content_type\"] == \"mixed\") for k in KEYS]\n"
    "table_sums   = [sum(c[\"tableCount\"] for c in chunks[k]) for k in KEYS]\n"
    "x = np.arange(len(KEYS))\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))\n"
    "\n"
    "ax1.bar(x, prose_counts, label=\"prose\",  color=\"#4e79a7\")\n"
    "ax1.bar(x, mixed_counts, bottom=prose_counts, label=\"mixed\", color=\"#f28e2b\")\n"
    "ax1.set_xticks(x)\n"
    "ax1.set_xticklabels(KEYS, rotation=30, ha=\"right\")\n"
    "ax1.set_ylabel(\"Chunk Count\")\n"
    "ax1.set_title(\"Content Type Breakdown per Document (prose vs. mixed)\")\n"
    "ax1.legend()\n"
    "\n"
    "ax2.bar(x, table_sums, color=\"#59a14f\")\n"
    "ax2.set_xticks(x)\n"
    "ax2.set_xticklabels(KEYS, rotation=30, ha=\"right\")\n"
    "ax2.set_ylabel(\"Total tableCount Sum\")\n"
    "ax2.set_title(\"Total Tables Referenced Across Chunks per Document\")\n"
    "for i, v in enumerate(table_sums):\n"
    "    ax2.text(i, v + 0.3, str(v), ha=\"center\", va=\"bottom\", fontsize=9)\n"
    "\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"04_content_type_breakdown.png\")\n"
    "plt.show()\n"
    "plt.close(fig)"
))

# ── Step 5 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 5 — Keyword Frequency Analysis\n\n"
    "Keywords are extracted per chunk by the chunker pipeline. Three views:\n\n"
    "1. **Corpus-wide top 25** — topics that dominate the six filings.\n"
    "2. **Apple vs Google top 15** — whether vocabulary diverges (Apple → hardware;\n"
    "   Google → advertising/AI).\n"
    "3. **Item 1A year-over-year** — Risk Factor language shifts expose emerging themes,\n"
    "   e.g. AI/supply-chain risk vocabulary growing in later years."
))

cells.append(code(
    "all_kws    = [kw for key in KEYS for c in chunks[key] for kw in c.get(\"keywords\", [])]\n"
    "apple_kws  = [kw for key in KEYS if \"apple\"  in key for c in chunks[key] for kw in c.get(\"keywords\", [])]\n"
    "google_kws = [kw for key in KEYS if \"google\" in key for c in chunks[key] for kw in c.get(\"keywords\", [])]\n"
    "\n"
    "item1a = {co: {yr: Counter() for yr in YEARS} for co in COMPANIES}\n"
    "for co in COMPANIES:\n"
    "    for yr in YEARS:\n"
    "        for c in chunks[f\"{co}_{yr}\"]:\n"
    "            if c.get(\"item\") == \"Item 1A\":\n"
    "                item1a[co][yr].update(c.get(\"keywords\", []))\n"
    "\n"
    "fig = plt.figure(figsize=(16, 16))\n"
    "\n"
    "# ── Plot 1: top 25 overall ───────────────────────────────────────────────\n"
    "ax1 = fig.add_subplot(3, 1, 1)\n"
    "top25 = Counter(all_kws).most_common(25)\n"
    "words25, cnt25 = zip(*top25)\n"
    "xi25 = np.arange(len(words25))\n"
    "ax1.bar(xi25, cnt25, color=\"#4e79a7\")\n"
    "ax1.set_xticks(xi25)\n"
    "ax1.set_xticklabels(words25, rotation=45, ha=\"right\", fontsize=9)\n"
    "ax1.set_ylabel(\"Frequency\")\n"
    "ax1.set_title(\"Top 25 Keywords — Entire Corpus\")\n"
    "\n"
    "# ── Plot 2: Apple vs Google grouped bars ────────────────────────────────\n"
    "ax2 = fig.add_subplot(3, 1, 2)\n"
    "a_top = dict(Counter(apple_kws).most_common(15))\n"
    "g_top = dict(Counter(google_kws).most_common(15))\n"
    "union = sorted(set(list(a_top) + list(g_top)),\n"
    "               key=lambda w: -(a_top.get(w, 0) + g_top.get(w, 0)))[:15]\n"
    "xi = np.arange(len(union)); bw = 0.35\n"
    "ax2.bar(xi - bw/2, [a_top.get(k, 0) for k in union], bw, label=\"Apple\",  color=\"#4e79a7\")\n"
    "ax2.bar(xi + bw/2, [g_top.get(k, 0) for k in union], bw, label=\"Google\", color=\"#e15759\")\n"
    "ax2.set_xticks(xi)\n"
    "ax2.set_xticklabels(union, rotation=45, ha=\"right\", fontsize=9)\n"
    "ax2.set_title(\"Top 15 Keywords: Apple vs. Google\")\n"
    "ax2.set_ylabel(\"Frequency\")\n"
    "ax2.legend()\n"
    "\n"
    "# ── Plot 3: Item 1A keywords by year, split by company ──────────────────\n"
    "for col, co in enumerate(COMPANIES):\n"
    "    ax3 = fig.add_subplot(3, 2, 5 + col)\n"
    "    co_all = sum((item1a[co][yr] for yr in YEARS), Counter())\n"
    "    top15  = [w for w, _ in co_all.most_common(15)]\n"
    "    xi = np.arange(len(top15)); bw = 0.25\n"
    "    for i, yr in enumerate(YEARS):\n"
    "        yr_cnts = [item1a[co][yr].get(kw, 0) for kw in top15]\n"
    "        ax3.bar(xi + (i-1)*bw, yr_cnts, bw, label=str(yr),\n"
    "                color=YEAR_COLORS[yr], alpha=0.85)\n"
    "    ax3.set_xticks(xi)\n"
    "    ax3.set_xticklabels(top15, rotation=45, ha=\"right\", fontsize=8)\n"
    "    ax3.set_title(f\"Item 1A Risk Keywords \\u2014 {co.title()}\")\n"
    "    ax3.set_ylabel(\"Frequency\")\n"
    "    ax3.legend(fontsize=8)\n"
    "\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"05_keyword_analysis.png\")\n"
    "plt.show()\n"
    "plt.close(fig)"
))

# ── Step 6 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 6 — hasNumbers Flag Analysis\n\n"
    "The `hasNumbers` flag marks chunks that contain quantitative content (dollar figures, %).\n"
    "For a financial RAG system this matters: numeric questions (revenue, margins, guidance)\n"
    "should retrieve `hasNumbers=True` chunks.\n\n"
    "**Expected pattern**: mixed chunks (which contain tables) should carry numbers in nearly\n"
    "all cases. Any mixed chunk with `hasNumbers=False` is a potential labelling inconsistency\n"
    "— it would be missed by a `hasNumbers`-filtered retrieval query despite containing tables."
))

cells.append(code(
    "prose_hasnums, mixed_hasnums = [], []\n"
    "for k in KEYS:\n"
    "    prose_ck = [c for c in chunks[k] if c[\"content_type\"] == \"prose\"]\n"
    "    mixed_ck = [c for c in chunks[k] if c[\"content_type\"] == \"mixed\"]\n"
    "    prose_hasnums.append(\n"
    "        100 * sum(c[\"hasNumbers\"] for c in prose_ck) / max(1, len(prose_ck)))\n"
    "    mixed_hasnums.append(\n"
    "        100 * sum(c[\"hasNumbers\"] for c in mixed_ck) / max(1, len(mixed_ck)))\n"
    "\n"
    "x = np.arange(len(KEYS)); bw = 0.35\n"
    "fig, ax = plt.subplots(figsize=(12, 6))\n"
    "bars_p = ax.bar(x - bw/2, prose_hasnums, bw, label=\"prose\",  color=\"#4e79a7\")\n"
    "bars_m = ax.bar(x + bw/2, mixed_hasnums, bw, label=\"mixed\",  color=\"#f28e2b\")\n"
    "ax.set_xticks(x)\n"
    "ax.set_xticklabels(KEYS, rotation=30, ha=\"right\")\n"
    "ax.set_ylabel(\"% chunks with hasNumbers=True\")\n"
    "ax.set_ylim(0, 115)\n"
    "ax.set_title(\"hasNumbers Flag Rate: Prose vs. Mixed Chunks per Document\")\n"
    "ax.legend()\n"
    "for i, (p, m) in enumerate(zip(prose_hasnums, mixed_hasnums)):\n"
    "    ax.text(i - bw/2, p + 1.5, f\"{p:.0f}%\", ha=\"center\", va=\"bottom\", fontsize=8)\n"
    "    ax.text(i + bw/2, m + 1.5, f\"{m:.0f}%\", ha=\"center\", va=\"bottom\", fontsize=8)\n"
    "\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"06_numbers_flag_analysis.png\")\n"
    "plt.show()\n"
    "plt.close(fig)\n"
    "\n"
    "dq_mixed = [(k, c[\"chunk_id\"]) for k in KEYS\n"
    "            for c in chunks[k] if c[\"content_type\"] == \"mixed\" and not c[\"hasNumbers\"]]\n"
    "if dq_mixed:\n"
    "    print(f\"Mixed chunks with hasNumbers=False ({len(dq_mixed)} total):\")\n"
    "    for doc, cid in dq_mixed:\n"
    "        print(f\"  {doc}: {cid}\")\n"
    "else:\n"
    "    print(\"No mixed chunks with hasNumbers=False.\")"
))

# ── Step 7 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 7 — Cross-Year Consistency\n\n"
    "Filing length and chunking output should be roughly stable year-over-year within a company.\n"
    "Sudden spikes in chunk count or average token size suggest the filing expanded significantly\n"
    "(e.g. new AI-risk disclosures post-2023) or the chunker behaved inconsistently.\n\n"
    "Tracking four metrics across 2023→2024→2025 separates real filing growth from\n"
    "pipeline artefacts and highlights which company's documents are more stable to process."
))

cells.append(code(
    "metrics = [\n"
    "    (\"chunk_count\",         \"Chunk Count\"),\n"
    "    (\"avg_token_estimate\",  \"Avg Token Estimate\"),\n"
    "    (\"table_chunk_pct\",     \"Table Chunk %\"),\n"
    "    (\"content_block_count\", \"Content Block Count\"),\n"
    "]\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n"
    "\n"
    "for ax, (col, label) in zip(axes.flat, metrics):\n"
    "    for co in COMPANIES:\n"
    "        vals = [df.loc[df[\"document\"] == f\"{co}_{yr}\", col].values[0] for yr in YEARS]\n"
    "        ax.plot(YEARS, vals, marker=\"o\", linewidth=2,\n"
    "                label=co.title(), color=COMPANY_COLORS[co])\n"
    "        for yr, v in zip(YEARS, vals):\n"
    "            ax.annotate(f\"{v:.0f}\", (yr, v),\n"
    "                        textcoords=\"offset points\", xytext=(5, 5), fontsize=8)\n"
    "    ax.set_xticks(YEARS)\n"
    "    ax.set_title(label)\n"
    "    ax.set_xlabel(\"Year\")\n"
    "    ax.set_ylabel(label)\n"
    "    ax.legend()\n"
    "\n"
    "fig.suptitle(\"Cross-Year Trends: 2023 \\u2192 2024 \\u2192 2025\", fontsize=14)\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"07_cross_year_trends.png\")\n"
    "plt.show()\n"
    "plt.close(fig)"
))

# ── Step 8 ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "## Step 8 — Data Quality Summary\n\n"
    "Consolidates all programmatic anomalies into a single table.\n"
    "These issues don't mean the data is broken — a tiny chunk may be a single-sentence\n"
    "disclosure; a missing section may not exist in that year's filing.\n"
    "But each entry warrants manual review before deploying the RAG system."
))

cells.append(code(
    "issues = []\n"
    "\n"
    "# Chunks below enforced minimum\n"
    "for key in KEYS:\n"
    "    for c in chunks[key]:\n"
    "        if c[\"token_estimate\"] < 200:\n"
    "            issues.append({\"type\": \"tiny_chunk\", \"document\": key,\n"
    "                            \"chunk_id\": c[\"chunk_id\"],\n"
    "                            \"detail\":   f\"token_estimate={c['token_estimate']}\"})\n"
    "\n"
    "# Sections present in >=1 doc but absent in others\n"
    "all_items_set = set(c[\"item\"] for key in KEYS for c in chunks[key] if c.get(\"item\"))\n"
    "for key in KEYS:\n"
    "    present = set(c[\"item\"] for c in chunks[key] if c.get(\"item\"))\n"
    "    for item in all_items_set - present:\n"
    "        issues.append({\"type\": \"missing_section\", \"document\": key,\n"
    "                        \"chunk_id\": \"N/A\", \"detail\": f\"{item} has 0 chunks\"})\n"
    "\n"
    "# Mixed chunks without numbers flag\n"
    "for key in KEYS:\n"
    "    for c in chunks[key]:\n"
    "        if c[\"content_type\"] == \"mixed\" and not c[\"hasNumbers\"]:\n"
    "            issues.append({\"type\": \"mixed_no_numbers\", \"document\": key,\n"
    "                            \"chunk_id\": c[\"chunk_id\"], \"detail\": \"mixed but hasNumbers=False\"})\n"
    "\n"
    "# Chunk count >30% deviation from company average\n"
    "for co in COMPANIES:\n"
    "    co_counts = {yr: len(chunks[f\"{co}_{yr}\"]) for yr in YEARS}\n"
    "    co_avg = np.mean(list(co_counts.values()))\n"
    "    for yr, cnt in co_counts.items():\n"
    "        if abs(cnt - co_avg) / co_avg > 0.30:\n"
    "            issues.append({\"type\": \"count_deviation\", \"document\": f\"{co}_{yr}\",\n"
    "                            \"chunk_id\": \"N/A\",\n"
    "                            \"detail\": f\"count={cnt}, avg={co_avg:.0f} (>30% diff)\"})\n"
    "\n"
    "issues_df  = pd.DataFrame(issues)\n"
    "summary_df = issues_df.groupby(\"type\").size().reset_index(name=\"count\")\n"
    "\n"
    "print(\"=== Issue Summary ===\")\n"
    "print(summary_df.to_string(index=False))\n"
    "print()\n"
    "pd.set_option(\"display.max_colwidth\", 60)\n"
    "print(\"=== All Issues ===\")\n"
    "print(issues_df.to_string(index=False))\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(10, 3.5))\n"
    "ax.axis(\"off\")\n"
    "tbl = ax.table(\n"
    "    cellText=summary_df.values.tolist(),\n"
    "    colLabels=[\"Issue Type\", \"Count\"],\n"
    "    cellLoc=\"center\",\n"
    "    loc=\"center\",\n"
    ")\n"
    "tbl.auto_set_font_size(False)\n"
    "tbl.set_fontsize(11)\n"
    "tbl.scale(1, 2.8)\n"
    "for j in range(2):\n"
    "    tbl[0, j].set_facecolor(\"#2c3e50\")\n"
    "    tbl[0, j].set_text_props(color=\"white\", fontweight=\"bold\")\n"
    "ax.set_title(\"Data Quality Issue Summary\", fontsize=13, pad=10)\n"
    "save_fig(fig, \"08_data_quality_issues.png\")\n"
    "plt.show()\n"
    "plt.close(fig)"
))

# ── Assemble notebook JSON ─────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

with open("eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("eda.ipynb written successfully.")
