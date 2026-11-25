"""Report Agent: compiles all agent outputs into an analyst-style PDF.

collect() gathers content (uses the LLM for summary/outlook); render_pdf() is pure layout.
Sections: Executive Summary, Key Financials, Red Flags, Company Comparison, Outlook, Sources.
"""
import re
from datetime import date
from xml.sax.saxutils import escape

import db
import llm
from agents import comparison_agent, document_agent
from agents.extraction_agent import period_year

SUMMARY_SYSTEM = """You are a senior equity research analyst. Use ONLY the metrics and flags given.
Write plain text (no markdown). Cite figures with their page like (p.57). Do not add outside facts."""
OUTLOOK_SYSTEM = """You summarise management outlook strictly from numbered excerpts. Plain text, no markdown,
3-5 sentences, cite each claim with its label like [S2]. If excerpts contain no outlook, say
"The filing excerpts do not contain a clear forward-looking statement." """

SEV_COLOR = {"high": "#c0392b", "medium": "#e67e22", "low": "#7f8c8d"}


def safe(t) -> str:
    """XML-escape and force Latin-1 so default PDF fonts never show boxes."""
    t = str(t if t is not None else "")
    t = t.replace("\u20b9", "Rs.").replace("\u2013", "-").replace("\u2014", "-") \
         .replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"[*#`]+", "", t)
    return escape(t.encode("latin-1", "replace").decode("latin-1"))


def _fmt(m):
    v = m["value_num"]
    s = f"{v:,.2f}" if m["source"] == "computed" else (m["value"] or f"{v:,.2f}")
    return f'{s} {m["unit"] or ""}'.strip()


def collect(session_name: str, doc_ids: list, progress=None):
    say = progress or (lambda m: None)
    docs = [d for d in (db.get_document(i) for i in doc_ids) if d]
    data = {"title": session_name, "date": date.today().isoformat(), "companies": []}
    for d in docs:
        say(f'Report Agent: summarising {d["company"]}...')
        metrics, flags = db.get_metrics(d["id"]), db.get_flags(d["id"])
        ver = [m for m in metrics if m["verified"]]
        latest = comparison_agent.latest_metrics(d["id"])
        facts = "\n".join(f'- {n}: {_fmt(m)} ({m["period"]}, p.{m["page"]})' for n, m in latest.items()) or "(none)"
        fl = "\n".join(f'- [{f["severity"]}] {f["category"]}: {f["description"]} (p.{f["page"]})' for f in flags) or "(none)"
        summary = llm.generate(f'Company: {d["company"]}\nKEY METRICS:\n{facts}\nRED FLAGS:\n{fl}\n\n'
                               "Write a 4-6 sentence executive summary covering performance, balance-sheet strength "
                               "and the most important risks.", SUMMARY_SYSTEM).strip() if ver else \
            "No verified metrics could be extracted from this document."
        say(f'Report Agent: outlook for {d["company"]}...')
        hits = document_agent.retrieve_many(
            ["outlook guidance future growth strategy next year demand environment", "risks and opportunities ahead"],
            [d["id"]], k=3)[:5]
        src = {f"S{i + 1}": h for i, h in enumerate(hits)}
        raw = llm.generate("\n\n".join(f'[{s}] page {h["page"]}\n{h["text"]}' for s, h in src.items()),
                           OUTLOOK_SYSTEM).strip() if hits else "No outlook text available."
        outlook = re.sub(r"\[(S\d+)\]", lambda m: f'(p.{src[m.group(1)]["page"]})' if m.group(1) in src else "", raw)
        data["companies"].append({"doc": d, "summary": summary, "outlook": outlook,
                                  "metrics": sorted(ver, key=lambda m: (m["source"] == "computed", m["name"], -period_year(m["period"]))),
                                  "flags": flags})
    if len(docs) >= 2:
        say("Report Agent: building comparison...")
        df, units, winners = comparison_agent.build_table(doc_ids)
        data["comparison"] = {"table": comparison_agent.table_markdown(df, units), "df": df, "units": units,
                              "winners": winners, "text": comparison_agent.commentary(doc_ids)}
    return data


def render_pdf(data: dict, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#1f3a5f"), spaceBefore=14, keepWithNext=1)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#2c5282"), keepWithNext=1)
    B = ParagraphStyle("B", parent=ss["BodyText"], leading=14)
    S = ParagraphStyle("S", parent=ss["BodyText"], fontSize=8, leading=10)
    P = lambda t, st=S: Paragraph(safe(t), st)

    def table(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b8c4")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")])]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f"))]
        t.setStyle(TableStyle(st))
        return t

    hd = lambda t: Paragraph(f'<font color="white"><b>{safe(t)}</b></font>', S)
    story = [Spacer(1, 3 * cm), Paragraph(safe("Equity Research Report"), ss["Title"]),
             Paragraph(safe(data["title"]), ParagraphStyle("sub", parent=ss["Heading2"], alignment=1)),
             Paragraph(safe(f'Generated {data["date"]} by the Multi-Agent Financial Research System'),
                       ParagraphStyle("c", parent=B, alignment=1)), Spacer(1, 1 * cm),
             Paragraph("<b>Companies covered</b>", B)]
    for c in data["companies"]:
        story.append(P(f'{c["doc"]["company"]}  -  {c["doc"]["filename"]}  ({c["doc"]["n_pages"]} pages)', B))
    story.append(Spacer(1, 1 * cm))
    story.append(P("Every figure in this report is traceable to a page and a verbatim quote in the source filing. "
                   "Ratios marked 'computed' are calculated from verified figures. This report is educational "
                   "and not investment advice.", S))
    story.append(PageBreak())

    story.append(Paragraph("1. Executive Summary", H1))
    for c in data["companies"]:
        story += [Paragraph(safe(c["doc"]["company"]), H2), P(c["summary"], B), Spacer(1, 6)]

    story.append(Paragraph("2. Key Financials", H1))
    for c in data["companies"]:
        story.append(Paragraph(safe(c["doc"]["company"]), H2))
        rows = [[hd(h) for h in ("Metric", "Period", "Value", "Page", "Basis")]]
        for m in c["metrics"]:
            rows.append([P(m["name"]), P(m["period"]), P(_fmt(m)), P(m["page"] or "-"),
                         P("computed" if m["source"] == "computed" else "reported")])
        story.append(table(rows, [5 * cm, 2.2 * cm, 5.3 * cm, 1.5 * cm, 2.5 * cm]) if len(rows) > 1
                     else P("No verified metrics extracted.", B))
        story.append(Spacer(1, 8))

    story.append(Paragraph("3. Red Flags", H1))
    for c in data["companies"]:
        story.append(Paragraph(safe(c["doc"]["company"]), H2))
        if not c["flags"]:
            story.append(P("No red flags detected by the automated checks.", B))
            continue
        rows = [[hd(h) for h in ("Severity", "Category", "Finding", "Page")]]
        for f in c["flags"]:
            col = SEV_COLOR.get(f["severity"], "#7f8c8d")
            rows.append([Paragraph(f'<font color="{col}"><b>{safe(f["severity"].upper())}</b></font>', S),
                         P(f["category"]), P(f["description"] + (f' "{f["quote"][:140]}"' if f["quote"] and f["source"] == "llm" else "")),
                         P(f["page"] or "-")])
        story.append(table(rows, [1.8 * cm, 3.5 * cm, 10 * cm, 1.4 * cm]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("4. Company Comparison", H1))
    comp = data.get("comparison")
    if comp:
        df = comp["df"]
        rows = [[hd("Metric")] + [hd(c) for c in df.columns]]
        for name, r in df.iterrows():
            rows.append([P(name)] + [P("n/a" if v != v else f'{v:,.2f} {comp["units"].get(name, {}).get(col, "")}')
                                     for col, v in r.items()])
        w = 17 * cm
        story.append(table(rows, [4 * cm] + [(w - 4 * cm) / max(len(df.columns), 1)] * len(df.columns)))
        story += [Spacer(1, 6), P(comp["text"], B)]
        if comp["winners"]:
            story.append(P("Ratio leaders: " + "; ".join(f"{k}: {v}" for k, v in comp["winners"].items()), S))
    else:
        story.append(P("Add at least two company documents to the session to enable benchmarking.", B))

    story.append(Paragraph("5. Outlook", H1))
    for c in data["companies"]:
        story += [Paragraph(safe(c["doc"]["company"]), H2), P(c["outlook"], B)]

    story += [Paragraph("6. Methodology & Sources", H1),
              P("Pipeline: Document Agent (parse, chunk, embed) -> Extraction Agent (LLM extraction with verbatim "
                "quotes, code-verified) -> Red Flag Agent (rules + verified text scan) -> Comparison / Research / "
                "Report agents. Unverified model outputs are excluded from this report.", B)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title=safe(data["title"])).build(story)
    return path


def generate_report(session_name, doc_ids, out_path, progress=None):
    return render_pdf(collect(session_name, doc_ids, progress), out_path)
