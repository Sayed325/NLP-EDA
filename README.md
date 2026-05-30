# RAG EDA — Apple & Google 10-K Filings (2023–2025)

Exploratory data analysis of a financial RAG corpus built from Apple and Google annual 10-K filings.
Six documents (3 years × 2 companies) are parsed into structured JSON and chunked for retrieval.
The EDA characterises chunk quality, section coverage, keyword trends, and data anomalies to inform
retrieval pipeline decisions before deployment.

---

## Folder Structure

```
RAG EDA/
├── data/
│   ├── apple_2023.json          # Parsed 10-K structure
│   ├── apple_2023_chunks.json   # Chunker output
│   ├── apple_2024.json / _chunks.json
│   ├── apple_2025.json / _chunks.json
│   ├── google_2023.json / _chunks.json
│   ├── google_2024.json / _chunks.json
│   └── google_2025.json / _chunks.json
├── output/                      # All generated PNG plots
├── eda.ipynb                    # Main analysis notebook
├── eda_utils.py                 # Shared helper functions (loaders, stats, save_fig)
├── run_eda.py                   # Headless runner — generates all plots without Jupyter
├── create_notebook.py           # Regenerates eda.ipynb from source strings
├── README.md
└── CONTEXT.md                   # Full project handoff log
```

---

## How to Run

**Interactive (Jupyter):**
```bash
jupyter notebook eda.ipynb
```
Run all cells top-to-bottom. Plots are saved to `output/` automatically.

**Headless (no Jupyter required):**
```bash
python run_eda.py
```
Executes every analysis step and writes all 8 PNGs to `output/`.

**Dependencies:**
```bash
pip install pandas matplotlib seaborn numpy jupyter nbformat
```

---

## Analysis Sections

| Step | Output | Description |
|------|--------|-------------|
| 1 | `01_corpus_overview.png` | Per-document stats table: sections, blocks, chunks, token summary, table coverage |
| 2 | `02_token_distribution.png` | Token-size histograms by year with min/target threshold markers |
| 3 | `03_section_heatmap.png` | Chunk count per 10-K item × document — reveals retrieval blind spots |
| 4 | `04_content_type_breakdown.png` | Prose vs. mixed chunk split and total table references per document |
| 5 | `05_keyword_analysis.png` | Top keywords corpus-wide, Apple vs. Google, and Item 1A year-over-year shift |
| 6 | `06_numbers_flag_analysis.png` | `hasNumbers` flag rate for prose vs. mixed chunks; flags mislabelled mixed chunks |
| 7 | `07_cross_year_trends.png` | 2023→2024→2025 line plots for chunk count, avg tokens, table %, content blocks |
| 8 | `08_data_quality_issues.png` | Summary table of all programmatic anomalies found in the corpus |

---

## Key Findings

- **Chunk count**: [See Section 7] Apple's 2024 filing produced significantly more chunks than 2023/2025.
- **Token distribution**: [See Section 2] Both companies have a cluster of tiny chunks (< 200 tokens) concentrated in boilerplate Items 10–14.
- **Section coverage**: [See Section 3] `apple_2023` has zero chunks for `Item 1C` (Cybersecurity) — the SEC rule requiring this item did not apply to Apple's FY2023 filing date.
- **Mixed chunk labelling**: [See Section 6] 14 mixed-type chunks across all documents are flagged `hasNumbers=False`, a potential filter inconsistency.
- **Keyword divergence**: [See Section 5] Apple's corpus is dominated by product-line terms; Google's by advertising and services vocabulary.
- **Data quality**: [See Section 8] 47 tiny chunks, 1 missing section, 14 mislabelled mixed chunks — none are blockers but each warrants review.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | DataFrames for stats and issue tables |
| `matplotlib` | Plot rendering |
| `seaborn` | Heatmap (Step 3), theme |
| `numpy` | Token statistics |
| `jupyter` | Interactive notebook execution |
| `nbformat` | Notebook file generation (`create_notebook.py`) |
