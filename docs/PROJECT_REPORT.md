# Project Report — Multi-Agent Financial Research System

## 1. Problem & objective
Finance students and early-career analysts spend hours reading 200+ page annual reports to extract
metrics, spot risks and compare peers. This project automates that workflow with specialised
agents while guaranteeing every output is traceable to the source document.

## 2. System overview
See `ARCHITECTURE.md` for the flow diagram. Upload triggers a sequential pipeline
(Document -> Extraction -> Red Flag); Research, Comparison and Report agents run on demand.

## 3. Agents
**Document Agent.** PyMuPDF parses each PDF page by page; text is cleaned and split into ~1000-char
page-aware chunks with 150-char overlap (chunks never cross pages, so each has an exact page number).
Chunks are embedded with Gemini `gemini-embedding-001` and stored in ChromaDB (cosine) with metadata
(doc_id, company, filename, page). SHA-256 hashing prevents re-embedding duplicates.

**Extraction Agent.** 20 target metrics in 4 groups (income statement, balance sheet, cash flow, KPIs).
For each group it retrieves relevant chunks, asks Gemini (temperature 0, JSON output) for current and
prior-period values plus a verbatim quote, then *verifies in code* that the quote exists in the cited
chunk and contains the number. Unverified rows are labelled and excluded from downstream analysis.
Ratios (net margin, D/E, current ratio, ROE, FCF) are computed in Python — never by the LLM.

**Red Flag Agent.** (a) Rule engine over verified metrics: falling revenue/profit, net loss, margin
compression, rising debt, high leverage, liquidity < 1, negative operating cash flow, weak earnings
quality, cash decline — each with severity thresholds. (b) Text scan over 8 risk themes (auditor
qualification, going concern, litigation, related-party, restatement, impairment, governance, internal
controls); an LLM flag is kept only if its quote is verified verbatim in the source.

**Comparison Agent.** Builds a metric x company table from latest-period verified values, identifies
the leader per ratio (direction-aware), and asks the LLM for commentary using only that table.

**Research Agent.** Plan (decompose into <=4 sub-questions, resolving follow-up pronouns) -> retrieve
(per-document retrieval so every company is represented) -> reason step by step with `[S#]` labels ->
post-process: invalid citations removed, labels converted to `[Company, p.N]`, ungrounded answers
flagged in the UI. Questions with no evidence return "Not found in the provided documents."

**Report Agent.** Collects outputs, generates grounded executive summaries and outlook, and renders a
PDF (ReportLab): Executive Summary, Key Financials, Red Flags, Company Comparison, Outlook, Methodology.

## 4. Strict source grounding — how it is enforced
1. Prompts contain only retrieved excerpts; model told to refuse otherwise.
2. Mandatory verbatim quote + page for every extracted number / flag.
3. Programmatic verification (quote-in-chunk, number-in-quote); failures excluded.
4. Arithmetic done in code, not by the LLM.
5. Citation validation in the Research Agent.

## 5. Testing
- `tests/test_offline.py`: 14 unit/integration tests with a stubbed LLM covering chunking, period
  normalisation, quote verification (including hallucinated quotes), ratio computation, every red-flag
  rule (troubled vs healthy company), text-flag filtering, citation mapping, comparison, PDF generation.
- `scripts/e2e_check.py`: runs the whole pipeline on seed documents, checks metric coverage, verification
  rate, red flags, grounded answers, comparison, PDF; reports accuracy vs `tests/golden.json`.
- Record your own results here after running (fill in): extraction accuracy __%, verification rate __%,
  average report generation time __ min, issues found/fixed __.

## 6. Optimisation notes
Batched query embeddings (one API call per agent step); metric grouping (4 LLM calls per document);
rate-limit spacing and exponential back-off; per-document retrieval for multi-company questions;
tolerant JSON parsing with one automatic retry; DB migrations so older databases keep working.

## 7. Limitations & future work
No OCR for scanned PDFs; table-heavy statements can be split across chunks; Indian filings quote in
INR crore while US filings use USD millions (units are shown, not converted); LLM quality depends on the
free-tier model. Future: OCR, table-aware parsing, currency normalisation, XBRL ingestion, charts in PDF.
