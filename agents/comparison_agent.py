"""Comparison Agent: side-by-side benchmarking built ONLY from verified extracted metrics."""
import pandas as pd

import db
import llm
from agents.extraction_agent import period_year

ABSOLUTE = ["Revenue", "Operating Profit", "EBITDA", "Net Profit", "EPS", "Total Assets", "Total Equity",
            "Total Debt", "Cash and Cash Equivalents", "Operating Cash Flow", "Free Cash Flow", "Employee Count"]
RATIOS = {"Net Margin %": True, "Operating Margin %": True, "Return on Equity %": True,
          "Debt-to-Equity": False, "Current Ratio": True}      # name -> higher_is_better
ORDER = ABSOLUTE + list(RATIOS)

SYSTEM = """You are a peer-benchmarking analyst. Use ONLY the table provided. Do not use outside
knowledge. Absolute figures may be in different currencies/units, so compare absolute values only
when units match; ratios are always comparable. Plain text, no markdown, 4-6 sentences."""


def labels_for(docs):
    names = [d["company"] for d in docs]
    return {d["id"]: (d["company"] if names.count(d["company"]) == 1 else f'{d["company"]} ({d["id"][:4]})')
            for d in docs}


def latest_metrics(doc_id):
    """Latest-period verified value per metric for one document."""
    rows = [m for m in db.get_metrics(doc_id) if m["verified"] and m["value_num"] is not None]
    latest = {}
    for r in rows:
        cur = latest.get(r["name"])
        if cur is None or period_year(r["period"]) > period_year(cur["period"]):
            latest[r["name"]] = r
    return latest


def build_table(doc_ids):
    """Returns (DataFrame indexed by metric, units dict, winners dict)."""
    docs = [d for d in (db.get_document(i) for i in doc_ids) if d]
    lab = labels_for(docs)
    data, units = {}, {}
    for d in docs:
        lm = latest_metrics(d["id"])
        col = {}
        for name, r in lm.items():
            col[name] = r["value_num"]
            units.setdefault(name, {})[lab[d["id"]]] = f'{r["unit"]} ({r["period"]})'
        data[lab[d["id"]]] = col
    df = pd.DataFrame(data).reindex([m for m in ORDER if any(m in c for c in data.values())])
    winners = {}
    for name, hib in RATIOS.items():
        if name in df.index and df.loc[name].notna().sum() >= 2:
            s = df.loc[name].dropna()
            winners[name] = s.idxmax() if hib else s.idxmin()
    return df, units, winners


def table_markdown(df, units):
    lines = ["| Metric | " + " | ".join(df.columns) + " |", "|---|" + "---|" * len(df.columns)]
    for name, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            cells.append("n/a" if pd.isna(v) else f"{v:,.2f} {units.get(name, {}).get(col, '')}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def commentary(doc_ids):
    df, units, winners = build_table(doc_ids)
    if df.shape[1] < 2 or df.empty:
        return "Need at least two analysed companies with extracted metrics to compare."
    md = table_markdown(df, units)
    best = "; ".join(f"best {k}: {v}" for k, v in winners.items())
    return llm.generate(f"TABLE:\n{md}\n\nComputed leaders: {best}\n\nWrite the peer comparison.", SYSTEM).strip()
