"""
Standalone runner — executes every analysis step and saves all 8 output PNGs.
Equivalent to running eda.ipynb top-to-bottom; useful for CI / headless environments.
"""
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from eda_utils import (
    DATA_DIR, OUTPUT_DIR, KEYS, COMPANIES, YEARS,
    YEAR_COLORS, COMPANY_COLORS,
    load_all, compute_corpus_stats, save_fig,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
sns.set_theme(style="whitegrid", font_scale=1.0)

docs, chunks = load_all()
total = sum(len(v) for v in chunks.values())
print(f"Loaded {len(docs)} documents — {total} total chunks\n")


# ── Step 1 — Corpus Overview ──────────────────────────────────────────────────
print("Step 1 — Corpus Overview")
df = compute_corpus_stats(docs, chunks)

fig, ax = plt.subplots(figsize=(15, 3.5))
ax.axis("off")
tbl = ax.table(
    cellText=df.values.tolist(),
    colLabels=df.columns.tolist(),
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 2.0)
for j in range(len(df.columns)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Corpus Overview — All Documents", fontsize=13, pad=10)
save_fig(fig, "01_corpus_overview.png")
plt.close(fig)
print(df.to_string(index=False))
print()


# ── Step 2 — Token Distribution ───────────────────────────────────────────────
print("Step 2 — Token Size Distribution")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, company in zip(axes, COMPANIES):
    for year in YEARS:
        key = f"{company}_{year}"
        tokens = [c["token_estimate"] for c in chunks[key]]
        tiny_pct = 100 * sum(t < 200 for t in tokens) / len(tokens)
        ax.hist(tokens, bins=30, alpha=0.6,
                label=f"{year}  (tiny: {tiny_pct:.0f}%)",
                color=YEAR_COLORS[year])
    ax.axvline(200, color="red",   linestyle="--", linewidth=1.4,
               label="min threshold (200)")
    ax.axvline(800, color="green", linestyle="--", linewidth=1.4,
               label="target window (800)")
    ax.set_title(f"{company.title()} — Token Distribution")
    ax.set_xlabel("Token Estimate")
    ax.set_ylabel("Chunk Count")
    ax.legend(fontsize=8)

fig.suptitle("Token Size Distribution by Company and Year", fontsize=13)
fig.tight_layout()
save_fig(fig, "02_token_distribution.png")
plt.close(fig)
print()


# ── Step 3 — Section Coverage Heatmap ────────────────────────────────────────
print("Step 3 — Section Coverage Heatmap")

def _item_key(x):
    m = re.match(r'Item\s+(\d+)(.*)', x)
    return (int(m.group(1)), m.group(2).strip()) if m else (999, x)

all_items = sorted(
    set(c["item"] for key in KEYS for c in chunks[key] if c.get("item")),
    key=_item_key,
)

matrix = pd.DataFrame(
    {key: Counter(c["item"] for c in chunks[key] if c.get("item")) for key in KEYS},
    index=all_items,
).fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    matrix, annot=True, fmt="d", cmap="YlOrRd",
    linewidths=0.4, linecolor="#cccccc",
    cbar_kws={"label": "Chunk Count"},
    ax=ax,
)
ax.set_title("Chunk Count by Section × Document", fontsize=14)
ax.set_xlabel("Document")
ax.set_ylabel("10-K Item")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
save_fig(fig, "03_section_heatmap.png")
plt.close(fig)
print()


# ── Step 4 — Content Type Breakdown ──────────────────────────────────────────
print("Step 4 — Content Type Breakdown")
prose_counts = [sum(1 for c in chunks[k] if c["content_type"] == "prose") for k in KEYS]
mixed_counts = [sum(1 for c in chunks[k] if c["content_type"] == "mixed") for k in KEYS]
table_sums   = [sum(c["tableCount"] for c in chunks[k]) for k in KEYS]
x = np.arange(len(KEYS))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

ax1.bar(x, prose_counts, label="prose",  color="#4e79a7")
ax1.bar(x, mixed_counts, bottom=prose_counts, label="mixed", color="#f28e2b")
ax1.set_xticks(x)
ax1.set_xticklabels(KEYS, rotation=30, ha="right")
ax1.set_ylabel("Chunk Count")
ax1.set_title("Content Type Breakdown per Document (prose vs. mixed)")
ax1.legend()

ax2.bar(x, table_sums, color="#59a14f")
ax2.set_xticks(x)
ax2.set_xticklabels(KEYS, rotation=30, ha="right")
ax2.set_ylabel("Total tableCount Sum")
ax2.set_title("Total Tables Referenced Across Chunks per Document")
for i, v in enumerate(table_sums):
    ax2.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=9)

fig.tight_layout()
save_fig(fig, "04_content_type_breakdown.png")
plt.close(fig)
print()


# ── Step 5 — Keyword Analysis ─────────────────────────────────────────────────
print("Step 5 — Keyword Analysis")
all_kws    = [kw for key in KEYS for c in chunks[key] for kw in c.get("keywords", [])]
apple_kws  = [kw for key in KEYS if "apple"  in key for c in chunks[key] for kw in c.get("keywords", [])]
google_kws = [kw for key in KEYS if "google" in key for c in chunks[key] for kw in c.get("keywords", [])]

item1a = {co: {yr: Counter() for yr in YEARS} for co in COMPANIES}
for co in COMPANIES:
    for yr in YEARS:
        for c in chunks[f"{co}_{yr}"]:
            if c.get("item") == "Item 1A":
                item1a[co][yr].update(c.get("keywords", []))

fig = plt.figure(figsize=(16, 16))

ax1 = fig.add_subplot(3, 1, 1)
top25 = Counter(all_kws).most_common(25)
words25, cnt25 = zip(*top25)
xi25 = np.arange(len(words25))
ax1.bar(xi25, cnt25, color="#4e79a7")
ax1.set_xticks(xi25)
ax1.set_xticklabels(words25, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Frequency")
ax1.set_title("Top 25 Keywords — Entire Corpus")

ax2 = fig.add_subplot(3, 1, 2)
a_top = dict(Counter(apple_kws).most_common(15))
g_top = dict(Counter(google_kws).most_common(15))
union = sorted(set(list(a_top) + list(g_top)),
               key=lambda w: -(a_top.get(w, 0) + g_top.get(w, 0)))[:15]
xi = np.arange(len(union)); bw = 0.35
ax2.bar(xi - bw/2, [a_top.get(k, 0) for k in union], bw, label="Apple",  color="#4e79a7")
ax2.bar(xi + bw/2, [g_top.get(k, 0) for k in union], bw, label="Google", color="#e15759")
ax2.set_xticks(xi)
ax2.set_xticklabels(union, rotation=45, ha="right", fontsize=9)
ax2.set_title("Top 15 Keywords: Apple vs. Google")
ax2.set_ylabel("Frequency")
ax2.legend()

for col, co in enumerate(COMPANIES):
    ax3 = fig.add_subplot(3, 2, 5 + col)
    co_all = sum((item1a[co][yr] for yr in YEARS), Counter())
    top15  = [w for w, _ in co_all.most_common(15)]
    xi = np.arange(len(top15)); bw = 0.25
    for i, yr in enumerate(YEARS):
        yr_cnts = [item1a[co][yr].get(kw, 0) for kw in top15]
        ax3.bar(xi + (i-1)*bw, yr_cnts, bw, label=str(yr),
                color=YEAR_COLORS[yr], alpha=0.85)
    ax3.set_xticks(xi)
    ax3.set_xticklabels(top15, rotation=45, ha="right", fontsize=8)
    ax3.set_title(f"Item 1A Risk Keywords — {co.title()}")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "05_keyword_analysis.png")
plt.close(fig)
print()


# ── Step 6 — hasNumbers Analysis ──────────────────────────────────────────────
print("Step 6 — hasNumbers Flag Analysis")
prose_hasnums, mixed_hasnums = [], []
for k in KEYS:
    prose_ck = [c for c in chunks[k] if c["content_type"] == "prose"]
    mixed_ck = [c for c in chunks[k] if c["content_type"] == "mixed"]
    prose_hasnums.append(100 * sum(c["hasNumbers"] for c in prose_ck) / max(1, len(prose_ck)))
    mixed_hasnums.append(100 * sum(c["hasNumbers"] for c in mixed_ck) / max(1, len(mixed_ck)))

x = np.arange(len(KEYS)); bw = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - bw/2, prose_hasnums, bw, label="prose",  color="#4e79a7")
ax.bar(x + bw/2, mixed_hasnums, bw, label="mixed",  color="#f28e2b")
ax.set_xticks(x)
ax.set_xticklabels(KEYS, rotation=30, ha="right")
ax.set_ylabel("% chunks with hasNumbers=True")
ax.set_ylim(0, 115)
ax.set_title("hasNumbers Flag Rate: Prose vs. Mixed Chunks per Document")
ax.legend()
for i, (p, m) in enumerate(zip(prose_hasnums, mixed_hasnums)):
    ax.text(i - bw/2, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.text(i + bw/2, m + 1.5, f"{m:.0f}%", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
save_fig(fig, "06_numbers_flag_analysis.png")
plt.close(fig)

dq_mixed = [(k, c["chunk_id"]) for k in KEYS
            for c in chunks[k] if c["content_type"] == "mixed" and not c["hasNumbers"]]
if dq_mixed:
    print(f"Mixed chunks with hasNumbers=False ({len(dq_mixed)}):")
    for doc, cid in dq_mixed:
        print(f"  {doc}: {cid}")
else:
    print("No mixed chunks with hasNumbers=False.")
print()


# ── Step 7 — Cross-Year Trends ────────────────────────────────────────────────
print("Step 7 — Cross-Year Consistency")
metrics = [
    ("chunk_count",         "Chunk Count"),
    ("avg_token_estimate",  "Avg Token Estimate"),
    ("table_chunk_pct",     "Table Chunk %"),
    ("content_block_count", "Content Block Count"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (col, label) in zip(axes.flat, metrics):
    for co in COMPANIES:
        vals = [df.loc[df["document"] == f"{co}_{yr}", col].values[0] for yr in YEARS]
        ax.plot(YEARS, vals, marker="o", linewidth=2,
                label=co.title(), color=COMPANY_COLORS[co])
        for yr, v in zip(YEARS, vals):
            ax.annotate(f"{v:.0f}", (yr, v),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xticks(YEARS)
    ax.set_title(label)
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.legend()

fig.suptitle("Cross-Year Trends: 2023 → 2024 → 2025", fontsize=14)
fig.tight_layout()
save_fig(fig, "07_cross_year_trends.png")
plt.close(fig)
print()


# ── Step 8 — Data Quality Summary ────────────────────────────────────────────
print("Step 8 — Data Quality Summary")
issues = []

for key in KEYS:
    for c in chunks[key]:
        if c["token_estimate"] < 200:
            issues.append({"type": "tiny_chunk", "document": key,
                           "chunk_id": c["chunk_id"],
                           "detail": f"token_estimate={c['token_estimate']}"})

all_items_set = set(c["item"] for key in KEYS for c in chunks[key] if c.get("item"))
for key in KEYS:
    present = set(c["item"] for c in chunks[key] if c.get("item"))
    for item in all_items_set - present:
        issues.append({"type": "missing_section", "document": key,
                       "chunk_id": "N/A", "detail": f"{item} has 0 chunks"})

for key in KEYS:
    for c in chunks[key]:
        if c["content_type"] == "mixed" and not c["hasNumbers"]:
            issues.append({"type": "mixed_no_numbers", "document": key,
                           "chunk_id": c["chunk_id"], "detail": "mixed but hasNumbers=False"})

for co in COMPANIES:
    co_counts = {yr: len(chunks[f"{co}_{yr}"]) for yr in YEARS}
    co_avg = np.mean(list(co_counts.values()))
    for yr, cnt in co_counts.items():
        if abs(cnt - co_avg) / co_avg > 0.30:
            issues.append({"type": "count_deviation", "document": f"{co}_{yr}",
                           "chunk_id": "N/A",
                           "detail": f"count={cnt}, avg={co_avg:.0f} (>30% diff)"})

issues_df  = pd.DataFrame(issues)
summary_df = issues_df.groupby("type").size().reset_index(name="count")

print("=== Issue Summary ===")
print(summary_df.to_string(index=False))
print()
print("=== All Issues (first 40) ===")
pd.set_option("display.max_colwidth", 60)
print(issues_df.head(40).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")
tbl = ax.table(
    cellText=summary_df.values.tolist(),
    colLabels=["Issue Type", "Count"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.8)
for j in range(2):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Data Quality Issue Summary", fontsize=13, pad=10)
save_fig(fig, "08_data_quality_issues.png")
plt.close(fig)

print("\nAll plots saved to output/")
print("Issues summary:")
print(summary_df.to_string(index=False))
