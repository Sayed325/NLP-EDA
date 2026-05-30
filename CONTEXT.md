# CONTEXT — RAG EDA Project Handoff

Generated: 2026-05-30  
Analyst: Claude Code (automated EDA generation)

---

## Files Created

| File | Purpose |
|------|---------|
| `eda_utils.py` | Shared helpers: `load_all()`, `compute_corpus_stats()`, `save_fig()`, constants (KEYS, COLORS, etc.) |
| `eda.ipynb` | Main Jupyter notebook — 18 cells (markdown + code), one section per analysis step |
| `run_eda.py` | Headless equivalent of the notebook; runs end-to-end and saves all 8 PNGs without Jupyter |
| `create_notebook.py` | Generates `eda.ipynb` from Python strings; re-run any time cell content changes |
| `README.md` | GitHub-facing project overview with folder layout, run instructions, findings summary |
| `CONTEXT.md` | This file — handoff reference |
| `output/` | Directory containing all 8 generated PNG plots |

---

## Output PNGs

| File | Description |
|------|-------------|
| `01_corpus_overview.png` | Dark-header table showing section_count, content_block_count, chunk_count, avg/min/max token, table_chunk_pct, tiny_chunk_count for all 6 docs |
| `02_token_distribution.png` | Overlaid histograms (3 years, alpha=0.6) per company with red dashed min=200 and green dashed target=800 threshold lines; legend shows % tiny per year |
| `03_section_heatmap.png` | 18-row × 6-col seaborn heatmap of chunk counts per 10-K Item × document; YlOrRd palette, annotated cells |
| `04_content_type_breakdown.png` | Stacked bar (prose/mixed per doc) + separate bar (total tableCount per doc) |
| `05_keyword_analysis.png` | 3-panel: top-25 corpus-wide, top-15 Apple vs Google grouped bars, top-15 Item 1A keywords by year for each company |
| `06_numbers_flag_analysis.png` | Grouped bars showing hasNumbers=True rate for prose vs. mixed chunks per doc, with % annotations |
| `07_cross_year_trends.png` | 2×2 line plots (Apple vs Google): chunk count, avg token estimate, table chunk %, content block count across 2023–2025 |
| `08_data_quality_issues.png` | Summary table (issue_type, count): tiny_chunk=47, mixed_no_numbers=14, missing_section=1 |

---

## Data Quality Issues Found

### Tiny Chunks (token_estimate < 200): 47 total

Pattern: concentrated in Items 10–14 (Directors, Compensation, Security Ownership, Related Transactions, Accountant Fees) and some boilerplate Items 1B/9C. Most extreme cases are 31 tokens — single-sentence statutory disclosures.

Apple (24 total across 3 years):
- `apple_2023`: chunks 0024, 0050–0056 (tokens: 31–185)
- `apple_2024`: chunks 0025, 0027, 0052, 0055–0058, 0120 (tokens: 31–163)
- `apple_2025`: chunks 0027, 0055–0061 (tokens: 31–126)

Google (23 total across 3 years):
- `google_2023`: chunks 0031, 0033–0034, 0090–0094 (tokens: 47–149)
- `google_2024`: chunks 0033, 0035–0036, 0090, 0094–0097 (tokens: 47–166)
- `google_2025`: chunks 0030 range + 0094 range (tokens: 47–)

**Recommendation**: filter or merge chunks below ~100 tokens before indexing; they add noise without retrieval value.

### Mixed Chunks with hasNumbers=False: 14 total

Chunks labelled `content_type=mixed` (implying embedded tables) but with `hasNumbers=False`. This is a labelling inconsistency — mixed chunks almost certainly contain numeric data and should be flagged.

| Document | chunk_id |
|----------|----------|
| apple_2023 | chunk_0035 |
| apple_2024 | chunk_0037 |
| apple_2025 | chunk_0035, chunk_0040 |
| google_2023 | chunk_0040, chunk_0062, chunk_0066 |
| google_2024 | chunk_0042, chunk_0064, chunk_0067 |
| google_2025 | chunk_0040, chunk_0049, chunk_0060, chunk_0063 |

**Recommendation**: inspect these 14 chunks manually; if they genuinely contain tables, the `hasNumbers` flag should be corrected in the chunker.

### Missing Section: 1 case

`apple_2023` has no chunks for `Item 1C` (Cybersecurity).

**Root cause**: The SEC adopted cybersecurity disclosure requirements (Item 1C) in July 2023, effective for fiscal years ending on or after December 15, 2023. Apple's FY2023 ended September 30, 2023 — before the effective date — so Apple was exempt from filing Item 1C that year. Apple 2024 and 2025 filings include Item 1C normally.

**Recommendation**: expected gap; no action required. Note it in any RAG metadata so retrieval filters that include Item 1C don't confuse the absent 2023 Apple section.

### Count Deviation: 0 cases

No company had a year with chunk count >30% from its 3-year average. (Apple's 2024 at 120 chunks vs avg ~95 is elevated but below the 30% threshold.)

---

## Chunk JSON Schema Reference

Top-level keys in each `*_chunks.json` file:

```
{
  "document_id":    str,
  "document_title": str,
  "schema":         str,
  "chunking":       dict,   # chunker config metadata
  "chunks":         list    # array of chunk objects (see below)
}
```

Each chunk object:

```
{
  "chunk_id":         str,         # e.g. "apple_2023::chunk_0001"
  "chunk_index":      int,         # 1-based index within document
  "text":             str,         # full chunk text
  "section_ids":      list[str],   # IDs of source sections (may repeat for multi-section chunks)
  "section_paths":    list[list],  # e.g. [["PART I", "Item 1"], ...]
  "section_titles":   list[str],   # full section titles
  "subsection_titles":list[str],
  "context_path":     list[str],   # e.g. ["PART I", "Item 1"]
  "part":             str,         # e.g. "PART I", "PART II"
  "item":             str|None,    # e.g. "Item 1", "Item 1A", "Item 8"
  "content_type":     str,         # "prose" | "mixed"
  "hasNumbers":       bool,        # True if chunk contains numeric data
  "keywords":         list[str],   # top keywords extracted by chunker
  "hasTables":        bool,        # True if any referenced section has tables
  "tableCount":       int,         # count of tables referenced in chunk
  "page_range":       null|list,   # always null in this dataset
  "subjects":         list[str],   # e.g. ["content_type:prose", "hasNumbers"]
  "token_estimate":   int,         # estimated token count
  "total_chunks":     int          # total chunks in this document
}
```

Parsed document JSON schema (`*_.json`):

```
{
  "document_title": str,
  "config":         dict,
  "sections": [
    {
      "id":             str,         # e.g. "ITEM_1", "ITEM_1A"
      "title":          str,
      "section_path":   list,
      "content_blocks": [
        {
          "id":               str,
          "level":            int,
          "title":            str,
          "section_path":     list,
          "text_content":     str,
          "heuristics_alerts":list,
          "nested_lists":     list,
          "images":           list
        }
      ],
      "tables": list
    }
  ]
}
```

---

## Assumptions Made

1. **Token estimate**: used `token_estimate` field from chunk JSON as-is; no independent tokenisation was performed.
2. **Tiny threshold**: 200 tokens is treated as the enforced minimum per the task brief. The chunker output contains chunks below this — assumed to be intentional pass-throughs for short statutory sections, not bugs.
3. **Section item key**: the `item` field (e.g. "Item 1A") is used for all section-level grouping. Chunks with `item=None` are excluded from section analysis; they are rare and appear in table-only pages.
4. **keyword frequency**: keywords are counted with multiplicity across chunks — a keyword appearing in 10 chunks is counted 10 times. This reflects retrieval relevance rather than document-level diversity.
5. **Content type labels**: `prose` and `mixed` are the only two values observed across all 578 chunks. No other types exist in this dataset.
6. **Cross-year deviation threshold**: 30% was chosen as a rule-of-thumb for "significant" change; no statistical baseline was available.
7. **`apple_2023` Item 1C absence**: confirmed as expected due to SEC filing date exemption, not a parsing error.
