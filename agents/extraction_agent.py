"""Extraction Agent: pulls key financial metrics from an indexed document.

Faithfulness safeguards
1. The LLM only sees retrieved excerpts and must return a verbatim quote.
2. Each row is VERIFIED in code: the quote must exist in the cited excerpt and
   the number must appear in the quote. Unverified rows are flagged, never trusted.
3. Ratios are COMPUTED in Python from verified numbers (no LLM arithmetic).
"""
import re

import db
import llm
from agents import document_agent

METRIC_GROUPS = {
    "Income statement": [
        ("Revenue", "total revenue from operations net sales"),
        ("Operating Profit", "operating profit EBIT profit from operations"),
        ("EBITDA", "EBITDA earnings before interest tax depreciation amortisation"),
        ("Net Profit", "net profit profit for the year attributable to owners of the company"),
        ("EPS", "basic earnings per share"),
        ("Interest Expense", "finance costs interest expense"),
    ],
    "Balance sheet": [
        ("Total Assets", "total assets consolidated balance sheet"),
        ("Total Equity", "total equity shareholders funds net worth"),
        ("Total Liabilities", "total liabilities"),
        ("Total Debt", "borrowings total debt loans"),
        ("Cash and Cash Equivalents", "cash and cash equivalents"),
        ("Current Assets", "total current assets"),
        ("Current Liabilities", "total current liabilities"),
    ],
    "Cash flow": [
        ("Operating Cash Flow", "net cash generated from operating activities"),
        ("Capital Expenditure", "purchase of property plant and equipment capital expenditure"),
        ("Free Cash Flow", "free cash flow"),
    ],
    "Other KPIs": [
        ("Operating Margin %", "operating margin percentage of revenue"),
        ("Return on Equity %", "return on equity ROE"),
        ("Dividend per Share", "dividend per share"),
        ("Employee Count", "number of employees headcount"),
    ],
}
ALL_METRICS = [m for g in METRIC_GROUPS.values() for m, _ in g]

SYSTEM = """You are a meticulous financial data extraction engine.
Use ONLY the numbered excerpts provided. Never use outside knowledge, never estimate,
never calculate. If a figure is not explicitly printed in the excerpts, omit it."""

PROMPT = """Extract these metrics: {names}

For each metric return the CURRENT period and, if shown, the PREVIOUS period.
Return a JSON array. Each item:
{{"metric": "<exact name from the list>", "period": "<e.g. FY2024>", "value": "<number exactly as printed>",
  "value_num": <number without commas>, "unit": "<e.g. INR crore, USD million, %, per share, count>",
  "chunk": "<excerpt id such as C3>", "quote": "<verbatim text copied from that excerpt, max 200 chars, containing the number>"}}

Prefer CONSOLIDATED figures. Negative numbers (brackets) must be negative in value_num.

EXCERPTS:
{context}
"""


# ---------- helpers ----------
def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def norm_period(p: str) -> str:
    """'FY24', '2023-24', 'March 31, 2024', 'Year ended 31 March 2024' -> 'FY2024'."""
    p = str(p or "")
    m = re.search(r"(20\d\d)\s*[-–/]\s*(\d{2,4})", p)
    if m:
        end = m.group(2)
        return f"FY{m.group(1)[:2] + end}" if len(end) == 2 else f"FY{end}"
    m = re.search(r"FY\s*'?(\d{2,4})", p, flags=re.I)
    if m:
        y = m.group(1)
        return f"FY20{y}" if len(y) == 2 else f"FY{y}"
    years = re.findall(r"20\d\d", p)
    return f"FY{years[-1]}" if years else (p.strip() or "N/A")


def period_year(p: str) -> int:
    m = re.search(r"(\d{4})", p or "")
    return int(m.group(1)) if m else 0


def verify(row: dict, chunk_text: str) -> bool:
    """Quote must appear in the excerpt, and the number must appear in the quote."""
    quote = norm_text(row.get("quote"))
    if len(quote) < 3 or quote not in norm_text(chunk_text):
        return False
    digits = re.sub(r"[^\d.]", "", str(row.get("value", "")).replace(",", ""))
    digits = digits.strip(".")
    return bool(digits) and digits in re.sub(r"[,\s]", "", quote)


def _to_float(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def _rows_from_items(items, ctx_map, metrics):
    rows = []
    for it in items if isinstance(items, list) else []:
        hit = ctx_map.get(str(it.get("chunk", "")).strip())
        name = it.get("metric")
        if not hit or name not in dict(metrics):
            continue
        num = it.get("value_num")
        num = num if isinstance(num, (int, float)) else _to_float(it.get("value"))
        rows.append({
            "name": name, "value": str(it.get("value", "")), "unit": it.get("unit", ""),
            "period": norm_period(it.get("period")), "page": hit["page"],
            "quote": it.get("quote", ""), "value_num": num,
            "verified": int(verify(it, hit["text"])), "source": "extracted"})
    return rows


# ---------- LLM extraction ----------
def extract_group(doc_id: str, group: str, metrics: list):
    queries = [f"{name} {hint}" for name, hint in metrics]
    hits = document_agent.retrieve_many(queries, [doc_id], k=3)[:16]
    if not hits:
        return []
    ctx_map = {f"C{i + 1}": h for i, h in enumerate(hits)}
    context = "\n\n".join(f"[{cid}] (page {h['page']})\n{h['text']}" for cid, h in ctx_map.items())
    names = ", ".join(n for n, _ in metrics)
    items = llm.generate_json(PROMPT.format(names=names, context=context), SYSTEM, default=[]) or []
    return _rows_from_items(items, ctx_map, metrics)


def extract_all(doc_id: str, progress=None):
    """Retrieve evidence by metric group, then extract all metrics in one LLM request."""
    say = progress or (lambda msg: None)
    hits_by_id = {}
    for group, metrics in METRIC_GROUPS.items():
        say(f"Extraction Agent: retrieving {group.lower()} passages...")
        queries = [f"{name} {hint}" for name, hint in metrics]
        for hit in document_agent.retrieve_many(queries, [doc_id], k=3)[:16]:
            hits_by_id.setdefault(hit["id"], hit)
    if not hits_by_id:
        return []

    ctx_map = {f"C{i + 1}": hit for i, hit in enumerate(hits_by_id.values())}
    context = "\n\n".join(f"[{cid}] (page {hit['page']})\n{hit['text']}"
                          for cid, hit in ctx_map.items())
    metrics = [metric for group in METRIC_GROUPS.values() for metric in group]
    names = ", ".join(name for name, _ in metrics)
    say("Extraction Agent: extracting all metric groups...")
    items = llm.generate_json(PROMPT.format(names=names, context=context), SYSTEM, default=[]) or []
    return _rows_from_items(items, ctx_map, metrics)


def dedupe(rows):
    best = {}
    for r in rows:
        key = (r["name"], r["period"])
        if key not in best or (r["verified"], r["value_num"] is not None) > \
                (best[key]["verified"], best[key]["value_num"] is not None):
            best[key] = r
    return list(best.values())


# ---------- deterministic ratios ----------
def compute_ratios(rows):
    vals = {(r["name"], r["period"]): r for r in rows if r["verified"] and r["value_num"] is not None}
    periods = sorted({p for (_, p) in vals}, key=period_year, reverse=True)
    out = []

    def add(name, period, num, unit, formula, parts):
        out.append({"name": name, "value": f"{num:.2f}", "unit": unit, "period": period,
                    "page": parts[0]["page"], "quote": formula, "value_num": round(num, 4),
                    "verified": 1, "source": "computed"})

    for p in periods:
        g = lambda n: vals.get((n, p))
        rev, np_, op, eq, debt = g("Revenue"), g("Net Profit"), g("Operating Profit"), g("Total Equity"), g("Total Debt")
        ca, cl, ocf = g("Current Assets"), g("Current Liabilities"), g("Operating Cash Flow")
        capex = g("Capital Expenditure")
        if rev and np_ and rev["value_num"]:
            add("Net Margin %", p, np_["value_num"] / rev["value_num"] * 100, "%", "Net Profit / Revenue x 100", [rev, np_])
        if rev and op and rev["value_num"] and not g("Operating Margin %"):
            add("Operating Margin %", p, op["value_num"] / rev["value_num"] * 100, "%", "Operating Profit / Revenue x 100", [rev, op])
        if debt and eq and eq["value_num"]:
            add("Debt-to-Equity", p, debt["value_num"] / eq["value_num"], "x", "Total Debt / Total Equity", [debt, eq])
        if ca and cl and cl["value_num"]:
            add("Current Ratio", p, ca["value_num"] / cl["value_num"], "x", "Current Assets / Current Liabilities", [ca, cl])
        if np_ and eq and eq["value_num"] and not g("Return on Equity %"):
            add("Return on Equity %", p, np_["value_num"] / eq["value_num"] * 100, "%", "Net Profit / Total Equity x 100 (closing equity)", [np_, eq])
        if ocf and capex and not g("Free Cash Flow"):
            add("Free Cash Flow", p, ocf["value_num"] - abs(capex["value_num"]), ocf["unit"],
                "Operating Cash Flow - |Capital Expenditure|", [ocf, capex])
    return out


def run(doc_id: str, progress=None):
    say = progress or (lambda m: None)
    rows = extract_all(doc_id, progress=say)
    rows = dedupe(rows)
    rows += compute_ratios(rows)
    db.replace_metrics(doc_id, rows)
    ok = sum(r["verified"] for r in rows)
    say(f"Extraction Agent: {len(rows)} metrics ({ok} verified)")
    return rows
