# Multi-Agent Financial Research System

Six cooperating AI agents read real company filings and produce grounded, cited analysis
and an analyst-style PDF report. Built for the Infosys Springboard internship.
Stack: Python, Streamlit, ChromaDB, SQLite, PyMuPDF, ReportLab, Google Gemini (free tier).

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your free GEMINI_API_KEY
```
Put 3-4 text-based annual-report PDFs (e.g. Infosys, TCS, Wipro, HCLTech from their investor
pages) in `data/seed/`, then:
```bash
streamlit run app.py
```
First launch indexes + analyses the seed docs (~5-15 min on the free tier; one-time). If the
free tier rate-limits you, raise `LLM_MIN_INTERVAL` in `.env` (e.g. 12).

## Using it
1. Create a **session** in the sidebar (seed documents attach automatically). Delete old sessions from **All workspaces**; their chat history is removed, while indexed PDFs remain available for comparison.
2. **Documents**: upload more PDFs; the Document -> Extraction -> Red Flag pipeline runs automatically.
3. **Analysis**: red flags and extracted metrics, each with page + verbatim quote and a verified badge.
4. **Comparison**: select any analyzed PDFs across workspaces for side-by-side benchmarking.
5. **Research Chat**: ask multi-part questions; see the reasoning plan and exact sources.
6. **Report**: generate and download the PDF.

## Agents
| Agent | File |
|---|---|
| Document | `agents/document_agent.py` |
| Extraction | `agents/extraction_agent.py` |
| Red Flag | `agents/red_flag_agent.py` |
| Comparison | `agents/comparison_agent.py` |
| Research | `agents/research_agent.py` |
| Report | `agents/report_agent.py` |
Orchestration: `orchestrator.py` · LLM wrapper: `llm.py` · data layer: `db.py` · UI: `app.py`

## Testing
```bash
python -m unittest tests.test_offline -v     # 14 offline tests, no API key needed
python scripts/e2e_check.py                  # full end-to-end run on your seed documents
```
For an extraction-accuracy score, copy `tests/golden_template.json` to `tests/golden.json`, fill in
5-8 numbers per company read from the PDFs, and re-run `e2e_check.py`.

Limitations: scanned (image-only) PDFs are not supported (no OCR); free-tier Gemini limits make
big documents slow; absolute figures across companies may use different units/currencies.
