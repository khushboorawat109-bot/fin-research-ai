"""Red Flag Agent: automatic anomaly detection.

Two complementary detectors
- RULES on verified metrics (YoY declines, margin compression, leverage, liquidity, cash quality)
- TEXT SCAN of the filing (auditor qualifications, going concern, litigation, restatements...)
  where every flag must carry a verbatim quote that is verified against the source excerpt.
"""
import db
import llm
from agents import document_agent
from agents.extraction_agent import norm_text, period_year

SCAN_QUERIES = {
    "Auditor qualification": "auditor's report qualified opinion adverse opinion disclaimer emphasis of matter basis for qualified opinion",
    "Going concern": "material uncertainty related to going concern substantial doubt ability to continue",
    "Litigation & contingencies": "contingent liabilities litigation claims tax disputes pending legal proceedings",
    "Related-party": "related party transactions loans advances to promoter entities",
    "Restatement": "restatement of previously issued financial statements prior period error correction",
    "Impairment / write-offs": "impairment of goodwill write-off exceptional items one-time charge",
    "Governance": "resignation of auditor director chief financial officer key managerial personnel",
    "Internal controls": "material weakness internal control over financial reporting deficiency",
}

SYSTEM = """You are a forensic financial analyst. Use ONLY the numbered excerpts.
Report a red flag ONLY if an excerpt explicitly shows a problem. Routine disclosures and clean
statements (e.g. an unmodified audit opinion) are NOT red flags. Never speculate."""

PROMPT = """Review the excerpts and list genuine red flags. Return a JSON array (empty [] if none).
Each item: {{"category": "<one of: {cats}>", "severity": "low|medium|high",
"description": "<one factual sentence>", "chunk": "<excerpt id like C2>",
"quote": "<verbatim text from that excerpt, max 200 chars>"}}

EXCERPTS:
{context}
"""


def _flag(cat, sev, desc, ref=None):
    return {"category": cat, "severity": sev, "description": desc,
            "page": ref["page"] if ref else None, "quote": (ref or {}).get("quote", ""),
            "verified": 1, "source": "rule"}


def rule_flags(metrics: list):
    """Pure function over verified metrics -> list of flags (unit-tested offline)."""
    by = {}
    for m in metrics:
        if m["verified"] and m["value_num"] is not None:
            by.setdefault(m["name"], {})[m["period"]] = m
    flags = []

    def pair(name):
        d = by.get(name, {})
        ps = sorted(d, key=period_year, reverse=True)
        return (d[ps[0]], d[ps[1]]) if len(ps) >= 2 else (None, None)

    def pct_change(cur, prev):
        return None if not prev["value_num"] else (cur["value_num"] - prev["value_num"]) / abs(prev["value_num"]) * 100

    for name, label in (("Revenue", "Revenue"), ("Net Profit", "Net profit")):
        cur, prev = pair(name)
        if cur:
            ch = pct_change(cur, prev)
            if ch is not None:
                if name == "Revenue" and ch < 0:
                    flags.append(_flag("Falling revenue", "high" if ch < -10 else "medium",
                                       f"{label} fell {abs(ch):.1f}% from {prev['period']} to {cur['period']}.", cur))
                if name == "Net Profit" and ch <= -15:
                    flags.append(_flag("Falling profit", "high" if ch <= -30 else "medium",
                                       f"{label} fell {abs(ch):.1f}% from {prev['period']} to {cur['period']}.", cur))
    np_cur = by.get("Net Profit", {})
    for p, m in np_cur.items():
        if m["value_num"] < 0:
            flags.append(_flag("Net loss", "high", f"Net loss reported for {p}.", m))

    for name in ("Net Margin %", "Operating Margin %"):
        cur, prev = pair(name)
        if cur:
            drop = prev["value_num"] - cur["value_num"]
            if drop >= 2:
                flags.append(_flag("Margin compression", "high" if drop >= 5 else "medium",
                                   f"{name[:-2]} fell {drop:.1f} pp ({prev['value_num']:.1f}% -> {cur['value_num']:.1f}%).", cur))

    cur, prev = pair("Total Debt")
    if cur:
        ch = pct_change(cur, prev)
        if ch is not None and ch >= 20:
            flags.append(_flag("Rising debt", "high" if ch >= 50 else "medium",
                               f"Total debt rose {ch:.1f}% from {prev['period']} to {cur['period']}.", cur))
    for p, m in by.get("Debt-to-Equity", {}).items():
        if m["value_num"] > 1.5 and p == max(by["Debt-to-Equity"], key=period_year):
            flags.append(_flag("High leverage", "high" if m["value_num"] > 3 else "medium",
                               f"Debt-to-equity is {m['value_num']:.2f}x in {p}.", m))
    for p, m in by.get("Current Ratio", {}).items():
        if m["value_num"] < 1 and p == max(by["Current Ratio"], key=period_year):
            flags.append(_flag("Liquidity risk", "high" if m["value_num"] < 0.7 else "medium",
                               f"Current ratio is {m['value_num']:.2f} in {p} (<1).", m))

    ocf = by.get("Operating Cash Flow", {})
    if ocf:
        latest = ocf[max(ocf, key=period_year)]
        if latest["value_num"] < 0:
            flags.append(_flag("Negative operating cash flow", "high",
                               f"Operating cash flow was negative in {latest['period']}.", latest))
        np_latest = np_cur.get(latest["period"])
        if np_latest and np_latest["value_num"] > 0 and latest["value_num"] < 0.5 * np_latest["value_num"]:
            flags.append(_flag("Earnings quality", "medium",
                               f"Operating cash flow is under half of net profit in {latest['period']}.", latest))
    cur, prev = pair("Cash and Cash Equivalents")
    if cur:
        ch = pct_change(cur, prev)
        if ch is not None and ch <= -30:
            flags.append(_flag("Cash decline", "medium",
                               f"Cash fell {abs(ch):.1f}% from {prev['period']} to {cur['period']}.", cur))
    return flags


def text_flags(doc_id: str):
    hits = document_agent.retrieve_many(list(SCAN_QUERIES.values()), [doc_id], k=2)[:16]
    if not hits:
        return []
    ctx_map = {f"C{i + 1}": h for i, h in enumerate(hits)}
    context = "\n\n".join(f"[{cid}] (page {h['page']})\n{h['text']}" for cid, h in ctx_map.items())
    items = llm.generate_json(PROMPT.format(cats=" | ".join(SCAN_QUERIES), context=context),
                              SYSTEM, default=[]) or []
    out = []
    for it in items if isinstance(items, list) else []:
        hit = ctx_map.get(str(it.get("chunk", "")).strip())
        quote = it.get("quote", "")
        if not hit or len(norm_text(quote)) < 8 or norm_text(quote) not in norm_text(hit["text"]):
            continue    # drop anything we cannot verify against the source
        sev = it.get("severity") if it.get("severity") in ("low", "medium", "high") else "medium"
        out.append({"category": it.get("category", "Other"), "severity": sev,
                    "description": it.get("description", ""), "page": hit["page"],
                    "quote": quote, "verified": 1, "source": "llm"})
    return out


def run(doc_id: str, progress=None):
    say = progress or (lambda m: None)
    say("Red Flag Agent: rule checks on metrics...")
    flags = rule_flags(db.get_metrics(doc_id))
    say("Red Flag Agent: scanning text for auditor/legal/governance issues...")
    try:
        flags += text_flags(doc_id)
    except Exception as e:
        say(f"  ! text scan failed: {e}")
    db.replace_flags(doc_id, flags)
    say(f"Red Flag Agent: {len(flags)} flags")
    return flags
