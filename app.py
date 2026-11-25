"""FinResearch AI - website-style Streamlit app.
Pages: Home, All workspaces, Documents, Agent workflow, Passage search, Analysis, Compare, Research chat, Report.
Same agents / database as before: only the front end changed."""
import html as _html
import re
import time

import pandas as pd
import streamlit as st

import config
import db
import orchestrator
from agents import comparison_agent, document_agent, report_agent, research_agent

st.set_page_config(page_title="FinResearch AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

CSS = """
.block-container{padding-top:1.2rem;max-width:1200px}.stAppDeployButton{display:none}
[data-testid="stSidebar"]{background:#0B2A30}
[data-testid="stSidebar"] .stMarkdown *,[data-testid="stSidebar"] label p,[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{color:#CFE4E3!important}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:6px 10px;border-radius:8px;margin-bottom:2px}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#14656A}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *{color:#fff!important;font-weight:600}
.slogo{color:#fff;font-weight:800;font-size:1.25rem}.slogo b{color:#2BB5AE}
h1,h2,h3{letter-spacing:-.02em}
.top{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:.85rem;color:#5B7078;margin-bottom:.9rem}
.chip{display:inline-block;font-size:.76rem;padding:3px 11px;border:1px solid #E1E9ED;border-radius:99px;background:#fff;color:#33494E;margin:0 4px 4px 0}
.chip.ok{color:#0B6B3F;border-color:#9BD7B7;background:#EEF9F2}
.card{background:#fff;border:1px solid #E1E9ED;border-radius:12px;padding:18px;box-shadow:0 1px 2px rgba(15,42,51,.05)}
.hov{transition:.18s}.hov:hover{transform:translateY(-3px);border-color:#2BB5AE;box-shadow:0 10px 26px rgba(11,122,120,.14)}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card h4{margin:6px 0 4px;font-size:1.02rem}.card p{margin:0;font-size:.9rem;color:#46595F}
.ico{width:40px;height:40px;border-radius:10px;background:#E1F3F2;display:flex;align-items:center;justify-content:center;font-weight:800;color:#0B7A78}
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:26px;align-items:center;background:linear-gradient(135deg,#071d22,#0B3B40 60%,#0E7C7B);color:#fff;border-radius:20px;padding:38px;margin-bottom:22px}
.hero h1{font-size:2.5rem;line-height:1.12;margin:12px 0;color:#fff}.hero h1 span{color:#5EEAD4}.hero p{color:#D5EEEC;font-size:1.05rem;margin:0}
.hero .chip{background:#12474D;border-color:#12474D;color:#fff}
.mock{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.2);border-radius:14px;padding:16px;font-size:.85rem;line-height:1.7}
.mock .q{background:rgba(255,255,255,.17);border-radius:9px;padding:8px 11px;margin-bottom:9px}.mock .cite{color:#5EEAD4;background:none}
.cite{color:#0B6B69;font-weight:700;background:#E1F3F2;border-radius:6px;padding:0 6px;font-size:.78rem;white-space:nowrap}
@media(max-width:900px){.hero,.g3,.kpis,.trk{grid-template-columns:1fr!important}}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:.2rem 0 1rem}
.kpi{background:#fff;border:1px solid #E1E9ED;border-radius:12px;padding:12px 16px}.kpi b{font-size:1.6rem;display:block}.kpi i{font-style:normal;font-size:.78rem;color:#5B7078}
.lab{font-size:.68rem;letter-spacing:.09em;font-weight:700;color:#0B6B69;margin:2px 0 8px}
.b{display:inline-block;font-size:.75rem;font-weight:600;padding:2px 10px;border-radius:99px;white-space:nowrap}
.b.g{background:#DDF3E7;color:#0B6B3F}.b.a{background:#DCEBFA;color:#0B4F94;animation:pl 1.2s infinite}.b.q{background:#ECEFF1;color:#46565D}.b.r{background:#FBE3E6;color:#A61B2D}.b.y{background:#FDEFC8;color:#7A4B00}
@keyframes pl{50%{opacity:.55}}
.trk{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:.2rem 0 1rem}
.trk div{background:#fff;border:1px solid #E1E9ED;border-radius:10px;padding:9px 11px;font-size:.78rem}.trk div.cur{border:2px solid #0B7A78}.trk b{display:block;font-size:.85rem;margin-bottom:4px}
.stp{padding:7px 0;border-bottom:1px solid #EDF2F4}.stp:last-child{border:0}
.fl{border:1px solid #E1E9ED;border-left:6px solid #A61B2D;border-radius:10px;background:#fff;padding:10px 14px;margin-bottom:10px}.fl.medium{border-left-color:#B7791F}.fl.low{border-left-color:#7A8B8E}
.qt{font-style:italic;color:#46595F;font-size:.8rem}
.hit{border:1px solid #E1E9ED;border-radius:12px;padding:12px 16px;margin-bottom:10px;background:#fff}
.hit .bar{height:6px;border-radius:6px;background:#E4ECEF;overflow:hidden;width:130px;margin:4px 0 8px}.hit .bar i{display:block;height:100%;background:#0B7A78}
mark{background:#FDE68A;border-radius:3px;padding:0 2px}
.empty{border:2px dashed #B8C8CE;border-radius:14px;padding:2rem;text-align:center;color:#46595F;background:#fff}
.foot{color:#5B7078;font-size:.78rem;text-align:center;margin-top:2rem}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hj(s):
    return " ".join(x.strip() for x in s.splitlines() if x.strip())


def H(s):
    st.markdown(hj(s), unsafe_allow_html=True)


def esc(x):
    return _html.escape(str(x if x is not None else "")).replace("$", "&#36;")


def md(t):
    return str(t).replace("$", "\\$")


AG = [("Document Agent", "Turns a PDF into searchable passages"),
      ("Extraction Agent", "Finds 20 metrics and proves each with a quote"),
      ("Red Flag Agent", "Rules on numbers plus a text scan for warnings"),
      ("Comparison Agent", "Benchmarks companies on verified ratios"),
      ("Research Agent", "Plans, retrieves, reasons and cites"),
      ("Report Agent", "Compiles everything into a PDF")]
STEPS = [["Parse every page (PyMuPDF)", "Chunk by page, about 1000 characters", "Embed passages with Gemini", "Store in ChromaDB with page tags"],
         ["Income statement metrics", "Balance sheet metrics", "Cash flow metrics", "KPIs, then verify every quote in code"],
         ["7 rule checks on verified metrics", "Scan 8 risk themes in the text", "Keep only quotes found word for word", "Rank flags by severity"],
         ["Pick the latest verified period", "Build the side-by-side table", "Find the leader per ratio", "Write peer commentary on request"],
         ["Plan sub-questions", "Retrieve passages per company", "Reason step by step from sources", "Validate citations"],
         ["Executive summary", "Key financials and red flags", "Company comparison", "Outlook, methodology and sources"]]
PAGES = ["Home", "All workspaces", "Documents", "Agent workflow", "Passage search", "Analysis", "Compare", "Research chat", "Report"]
WS_PAGES = PAGES[2:]
SUGG = ["What was the total revenue and how did it change from last year?", "Summarise the main risks mentioned in the report.",
        "How have net profit and operating margin changed?", "Is the company's debt level a concern?"]
LBL = {"d": ("Done", "g"), "r": ("Working now", "a"), "q": ("Waiting", "q")}


def trk_html(states):
    return hj('<div class="trk">' + "".join(
        f'<div class="{"cur" if s == "r" else ""}"><b>{i + 1}. {AG[i][0].split()[0]}</b><span class="b {LBL[s][1]}">{LBL[s][0]}</span></div>'
        for i, s in enumerate(states)) + "</div>")


def kpi_html(items):
    return hj('<div class="kpis">' + "".join(f'<div class="kpi"><b>{n}</b><i>{l}</i></div>' for n, l in items) + "</div>")


def flag_html(f):
    how = "Rule on verified metrics" if f["source"] == "rule" else "Text scan of the filing"
    q = f'<div class="qt">“{esc(f["quote"][:220])}”</div>' if f["quote"] and f["source"] == "llm" else ""
    col = {"high": "#A61B2D", "medium": "#8A5A00"}.get(f["severity"], "#46565D")
    return hj(f'<div class="fl {esc(f["severity"])}"><b style="color:{col}">{esc(f["severity"]).upper()} · {esc(f["category"])}</b>'
              f'<div>{esc(f["description"])}</div><div class="qt">{how} · page {esc(f["page"])}</div>{q}</div>')


def risk_level(flags):
    s = sum(3 if f["severity"] == "high" else 1 if f["severity"] == "medium" else 0 for f in flags)
    return ("Low", "g") if s < 3 else ("Moderate", "y") if s < 6 else ("High", "r")


def ws_stats(wid):
    docs = db.session_documents(wid)
    an = [d for d in docs if d["status"] == "analyzed"]
    flags = [f for d in an for f in db.get_flags(d["id"])]
    ver = sum(1 for d in an for m in db.get_metrics(d["id"]) if m["verified"])
    return docs, an, flags, ver


# ---------- navigation callbacks ----------
def goto(p):
    st.session_state.page = p


def open_ws(wid, p="Agent workflow"):
    st.session_state.sid = wid
    st.session_state.ws_pick = wid
    st.session_state.page = p


def make_ws(name, p="Agent workflow"):
    open_ws(db.create_session(name), p)


def pick_changed():
    st.session_state.sid = st.session_state.ws_pick


def request_workspace_delete(wid):
    st.session_state.delete_workspace_id = wid


def confirm_workspace_delete():
    wid = st.session_state.get("delete_workspace_id")
    if not wid or not any(session["id"] == wid for session in db.list_sessions()):
        st.session_state.pop("delete_workspace_id", None)
        return
    db.delete_session(wid)
    if st.session_state.get("sid") == wid:
        del st.session_state["sid"]
    if st.session_state.get("ws_pick") == wid:
        st.session_state["ws_pick"] = None
    st.session_state.pop(f"report_{wid}", None)
    for key in list(st.session_state.keys()):
        if key.startswith(f"comp_{wid}_"):
            del st.session_state[key]
    st.session_state.pop("delete_workspace_id", None)


def cancel_workspace_delete():
    st.session_state.pop("delete_workspace_id", None)


# ---------- live pipeline (shows which agent is working right now) ----------
def live_pipeline(run_fn):
    tr, lg, lines, cur = st.empty(), st.empty(), [], [0]

    def paint(states):
        tr.markdown(trk_html(states), unsafe_allow_html=True)

    def cb(m):
        lines.append(m)
        if m.startswith("Red Flag"):
            cur[0] = 2
        elif m.startswith("Extraction"):
            cur[0] = max(cur[0], 1)
        paint(["d"] * cur[0] + ["r"] + ["q"] * (5 - cur[0]))
        lg.code("\n".join(lines[-10:]))

    paint(["r"] + ["q"] * 5)
    result = run_fn(cb)
    complete = result is not False
    if isinstance(result, dict):
        complete = result.get("status") == "analyzed"
    if complete:
        paint(["d", "d", "d", "q", "q", "q"])
    else:
        paint(["d"] * cur[0] + ["q"] * (6 - cur[0]))
    return complete


# ---------- start-up ----------
db.init_db()
if "seeded" not in st.session_state:
    if not config.GEMINI_API_KEY:
        st.error("Add GEMINI_API_KEY to a .env file (see .env.example), then restart.")
        st.stop()
    if any(f.suffix.lower() in (".pdf", ".txt") for f in config.SEED_DIR.iterdir()):
        with st.status("Preparing your seed company documents (first run takes a few minutes)...", expanded=True) as stt:
            orchestrator.bootstrap(progress=st.write)
            stt.update(label="Seed documents ready", state="complete", expanded=False)
    else:
        orchestrator.bootstrap()
    st.session_state.seeded = True
st.session_state.setdefault("page", "Home")

# ---------- sidebar ----------
sessions = db.list_sessions()
ids = [s["id"] for s in sessions]
names = {s["id"]: s["name"] for s in sessions}
st.sidebar.markdown('<div class="slogo">FinResearch<b> AI</b></div>', unsafe_allow_html=True)
st.sidebar.caption("Multi-agent financial research")
st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
st.sidebar.divider()
if ids:
    if st.session_state.get("ws_pick") not in ids:
        st.session_state.ws_pick = st.session_state.get("sid") if st.session_state.get("sid") in ids else ids[0]
    st.session_state.setdefault("sid", st.session_state.ws_pick)
    st.sidebar.selectbox("Active workspace", ids, key="ws_pick", format_func=names.get, on_change=pick_changed)
def create_from_form():
    name = st.session_state.get("new_ws_name", "").strip()
    if name:
        make_ws(name, "Documents")
        st.session_state.new_ws_name = ""


with st.sidebar.form("new_ws"):
    st.text_input("New workspace", placeholder="e.g. Indian IT peers FY24", key="new_ws_name")
    st.form_submit_button("➕ Create", use_container_width=True, on_click=create_from_form)

page = st.session_state.page
sid = st.session_state.get("sid") if st.session_state.get("sid") in ids else None
chips = f'<span class="chip ok">● Gemini connected</span><span class="chip">{len(sessions)} workspaces</span>'
H(f'<div class="top"><div>{esc(page)}{" · " + esc(names[sid]) if sid and page in WS_PAGES and page != "Compare" else ""}</div><div>{chips}</div></div>')

# =========================== HOME ===========================
if page == "Home":
    H('''<div class="hero"><div><span class="chip">Multi-agent AI</span><span class="chip">Free to run</span>
    <h1>Research any company filing <span>in minutes, not days.</span></h1>
    <p>Create a workspace, ingest annual reports, search every indexed passage and let six AI agents extract metrics,
    flag risks, compare peers and write the report, with a page citation for every claim.</p></div>
    <div class="mock"><div class="lab" style="color:#A8D8D3">EXAMPLE ANSWER</div><div class="q">Which company has the stronger balance sheet?</div>
    Step 1: Compare debt-to-equity.<br>Step 2: Alpha 0.20x vs Beta 0.80x.<br>
    <b>Answer:</b> Alpha is stronger <span class="cite">[Alpha Ltd, p.31]</span> <span class="cite">[Beta Ltd, p.28]</span></div></div>''')
    c1, c2, _ = st.columns([1.3, 1.3, 2])
    if ids:
        c1.button("Open my workspaces", type="primary", on_click=goto, args=("All workspaces",), use_container_width=True)
    else:
        c1.button("🚀 Create your first workspace", type="primary", on_click=make_ws, args=("My first research", "Documents"), use_container_width=True)
    c2.button("See the agents work", on_click=goto, args=("Agent workflow",), use_container_width=True, disabled=not ids)
    st.markdown("### Your six-agent research team")
    st.caption("Each agent has one job. Together they turn raw filings into verified insight.")
    H('<div class="g3">' + "".join(f'<div class="card hov"><div class="ico">{i + 1}</div><h4>{n}</h4><p>{d}</p></div>'
                                   for i, (n, d) in enumerate(AG)) + "</div>")
    st.markdown("### Numbers you can trust")
    H('''<div class="g3"><div class="card"><h4>Source-grounded</h4><p>The model sees only filing excerpts and must quote them.</p></div>
    <div class="card"><h4>Verified in code</h4><p>Each quote must exist on the cited page and contain the number.</p></div>
    <div class="card"><h4>Math in Python</h4><p>Margins and ratios are computed by code, never guessed.</p></div></div>''')
    H('<div class="foot">FinResearch AI · Multi-Agent Financial Research · Infosys Springboard Internship</div>')
    st.stop()

# =========================== ALL WORKSPACES ===========================
if page == "All workspaces":
    st.markdown("## Research workspaces")
    st.caption("Manage research workspaces, ingest documents and search indexed passages.")
    if not sessions:
        H('<div class="empty">No workspaces yet. Create one from the sidebar.</div>')
    for i in range(0, len(sessions), 3):
        for col, s in zip(st.columns(3), sessions[i:i + 3]):
            d_, an_, fl_, _v = ws_stats(s["id"])
            with col:
                H(f'<div class="card hov"><h4 style="font-size:1.1rem">{esc(s["name"])}</h4><p style="font-size:.78rem;margin-bottom:10px">Created {esc(s["created_at"][:10])}</p>'
                  f'<span class="chip">{len(d_)} docs</span><span class="chip">{len(an_)} analysed</span>'
                  f'<span class="chip">{sum(1 for f in fl_ if f["severity"] == "high")} high flags</span></div>')
                open_col, delete_col = st.columns([3, 1])
                open_col.button("Open", key=f"o{s['id']}", on_click=open_ws, args=(s["id"],), type="primary")
                delete_col.button("Delete", key=f"del{s['id']}", on_click=request_workspace_delete, args=(s["id"],))
    pending_delete = st.session_state.get("delete_workspace_id")
    if pending_delete in names:
        st.warning(f"Delete workspace '{names[pending_delete]}'?")
        st.caption("Its workspace and chat history will be removed. PDFs remain indexed and available to Compare.")
        confirm_col, cancel_col = st.columns(2)
        confirm_col.button("Confirm delete", key=f"confirm_del{pending_delete}", type="primary",
                           on_click=confirm_workspace_delete)
        cancel_col.button("Cancel", key=f"cancel_del{pending_delete}", on_click=cancel_workspace_delete)
    elif pending_delete is not None:
        st.session_state.pop("delete_workspace_id", None)
    st.stop()

# ---------- workspace pages need an active workspace ----------
if not sid:
    H('<div class="empty">Open or create a workspace first.</div>')
    st.button("Go to All workspaces", on_click=goto, args=("All workspaces",))
    st.stop()
current = next(s for s in sessions if s["id"] == sid)
docs, analysed, all_flags, n_ver = ws_stats(sid)
doc_ids = [d["id"] for d in docs]
messages = db.get_messages(sid)
st.markdown(f"## {current['name']}")
if page == "Compare":
    H(kpi_html([(len(db.analyzed_documents()), "Analyzed PDFs across workspaces")]))
else:
    H(kpi_html([(len(docs), "Documents"), (len(analysed), "Analysed"),
                (sum(1 for f in all_flags if f["severity"] == "high"), "High-severity flags"), (n_ver, "Verified metrics")]))


def empty(msg):
    H(f'<div class="empty">{msg}</div>')


def doc_label(d):
    return f"{d['company']} - {d['filename']}"


# =========================== DOCUMENTS ===========================
if page == "Documents":
    st.markdown("#### Ingest company filings")
    st.caption("Upload annual reports or 10-K filings. Text-based PDFs work best; scanned image PDFs are not supported.")
    up = st.file_uploader("Drop PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    if up and st.button("▶ Run the agent pipeline", type="primary"):
        for f in up:
            dest = config.UPLOAD_DIR / f.name
            dest.write_bytes(f.getbuffer())
            st.markdown(f"**{f.name}**")
            try:
                complete = live_pipeline(lambda cb: orchestrator.process_upload(dest, sid, progress=cb))
                if not complete:
                    st.error(f"{f.name}: analysis did not finish. Check the pipeline log and retry.")
            except Exception as e:
                st.error(f"{f.name}: {e}")
        st.rerun()
    st.markdown(f"#### Documents in this workspace ({len(docs)})")
    if not docs:
        empty("No documents yet. Upload a filing above to start the agents.")
    for d in docs:
        c1, c2, c3 = st.columns([5, 3, 1])
        c1.markdown(f"**{d['company']}**  \n`{d['filename']}`" + ("  🌱 seed" if d["is_seed"] else ""))
        c2.caption(f"{'✅ analysed' if d['status'] == 'analyzed' else '⚠️ not analysed'} · {d['n_pages']} pages · {d['n_chunks']} passages")
        if c3.button("Remove", key=f"rm{d['id']}"):
            db.detach_document(sid, d["id"])
            st.rerun()

# =========================== AGENT WORKFLOW ===========================
elif page == "Agent workflow":
    st.markdown("#### Agent workflow")
    st.caption("Select an agent to see its input, what it does and what it produced for your documents.")
    if not docs:
        empty("Add a document first. The agents start working as soon as a filing is uploaded.")
        st.button("Go to Documents", on_click=goto, args=("Documents",))
        st.stop()
    doc = st.selectbox("Document", docs, format_func=doc_label)
    metrics, flags = db.get_metrics(doc["id"]), db.get_flags(doc["id"])
    an_ok = doc["status"] == "analyzed"
    states = ["d", "d" if (an_ok or metrics) else "q", "d" if an_ok else "q", "d" if len(analysed) >= 2 else "q",
              "d" if any(m["role"] == "assistant" for m in messages) else "q", "d" if st.session_state.get(f"report_{sid}") else "q"]
    trk_slot = st.empty()
    trk_slot.markdown(trk_html(states), unsafe_allow_html=True)
    if st.button("▶ Re-run Extraction and Red Flag agents on this document"):
        live_pipeline(lambda cb: orchestrator.analyze(doc["id"], cb))
        st.rerun()
    left, right = st.columns([1, 2.2])
    mark = {"d": "✅", "r": "⏳", "q": "○"}
    pick = left.radio("Agent", range(6), format_func=lambda i: f"{mark[states[i]]}  {i + 1} · {AG[i][0]}", label_visibility="collapsed")
    with right:
        st.markdown(f"##### {AG[pick][0]}")
        inputs = [f"{doc['filename']} · {doc['n_pages']} pages", f"Indexed passages of {doc['company']}",
                  f"Verified metrics and passages of {doc['company']}", f"Verified metrics of {len(analysed)} analysed companies",
                  "Your question and the passages of all workspace documents", "Outputs of all five earlier agents"]
        H(f'<div class="lab">INPUT</div><div style="margin-bottom:10px">{esc(inputs[pick])}</div><div class="lab">PROCESSING</div>'
          + "".join(f'<div class="stp">{"✅" if states[pick] == "d" else "○"} {s}</div>' for s in STEPS[pick]))
        H('<div class="lab" style="margin-top:12px">OUTPUT</div>')
        if pick == 0:
            H(kpi_html([(doc["n_pages"], "Pages parsed"), (doc["n_chunks"], "Passages indexed"), (doc["id"], "Document id"), (doc["status"], "Status")]))
            try:
                r = document_agent._collection().get(where={"doc_id": doc["id"]}, limit=1, include=["documents", "metadatas"])
                if r["documents"]:
                    H(f'<div class="card"><span class="chip">{esc(doc["company"])}</span><span class="chip">page {esc(r["metadatas"][0]["page"])}</span>'
                      f'<p style="margin-top:6px">{esc(r["documents"][0][:380])}…</p></div>')
            except Exception:
                pass
        elif pick == 1:
            if not metrics:
                st.info("No metrics yet. Press the re-run button above.")
            else:
                ex = [m for m in metrics if m["source"] == "extracted"]
                H(kpi_html([(sum(m["verified"] for m in ex), "Verified"), (sum(1 for m in ex if not m["verified"]), "Rejected / unverified"),
                            (len(metrics) - len(ex), "Computed in Python"), (len(metrics), "Total metrics")]))
                df = pd.DataFrame(metrics)
                df["status"] = df.apply(lambda r: "computed" if r["source"] == "computed" else ("✅ verified" if r["verified"] else "⚠️ rejected"), axis=1)
                st.dataframe(df[["name", "period", "value", "unit", "page", "status", "quote"]].sort_values(["name", "period"]),
                             use_container_width=True, hide_index=True, height=300)
        elif pick == 2:
            lvl, col = risk_level(flags)
            H(f'<div style="margin-bottom:10px">Risk indicator: <span class="b {col}">{lvl}</span> · {len(flags)} flags</div>')
            if not flags:
                st.success("No red flags detected.") if an_ok else st.info("Not run yet.")
            H("".join(flag_html(f) for f in flags))
        elif pick == 3:
            if len(analysed) < 2:
                st.info("Needs at least two analysed companies in this workspace.")
            else:
                cdf, units, win = comparison_agent.build_table([d["id"] for d in analysed])
                disp = cdf.copy().astype(object)
                for n_, row in cdf.iterrows():
                    for c in cdf.columns:
                        disp.loc[n_, c] = "n/a" if pd.isna(row[c]) else f"{row[c]:,.2f} {units.get(n_, {}).get(c, '')}"
                st.dataframe(disp, use_container_width=True)
                if win:
                    st.markdown("**🏆 Leaders:** " + " · ".join(f"{k} → **{v}**" for k, v in win.items()))
        elif pick == 4:
            last = [m for m in messages if m["role"] == "assistant"]
            if not last:
                st.info("No questions asked yet. Use the Research chat page.")
            else:
                m = last[-1]
                st.markdown(md(m["content"]))
                meta = m.get("meta") or {}
                for i, p in enumerate(meta.get("plan", []), 1):
                    st.caption(md(f"Sub-question {i}: {p}"))
                for s_ in meta.get("sources", []):
                    st.caption(md(f"📚 {s_['label']} · {s_['filename']}"))
        else:
            path = st.session_state.get(f"report_{sid}")
            if path:
                with open(path, "rb") as fh:
                    st.download_button("⬇️ Download PDF report", fh.read(), file_name=f"{current['name']}_report.pdf", mime="application/pdf")
            else:
                st.info("Not generated yet. Use the Report page.")

# =========================== PASSAGE SEARCH ===========================
elif page == "Passage search":
    st.markdown("#### Search indexed passages")
    st.caption("Semantic search across every page of the documents in this workspace.")
    if not docs:
        empty("Ingest a document first, then search its passages here.")
    else:
        q1, q2 = st.columns([4, 1])
        qy = q1.text_input("Search", placeholder="e.g. employee attrition, auditor remarks, dividend policy", label_visibility="collapsed")
        k = q2.slider("Results", 3, 15, 6)
        scope = st.multiselect("Search in", doc_ids, default=doc_ids, format_func=lambda i: next(d["company"] for d in docs if d["id"] == i))
        if qy and scope:
            with st.spinner("Searching the vector index..."):
                try:
                    hits = document_agent.retrieve(qy, scope, k=k)
                except Exception as e:
                    hits = []
                    st.error(f"Search failed: {e}")
            st.caption(f"{len(hits)} passages found")
            words = sorted({w for w in re.findall(r"\w{3,}", qy)}, key=len, reverse=True)
            for h_ in hits:
                body = esc(h_["text"][:900].replace("\n", " "))
                for w in words:
                    body = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", body)
                pct = max(0, min(100, int(h_["score"] * 100)))
                H(f'<div class="hit"><div style="display:flex;justify-content:space-between"><b>{esc(h_["company"])} · page {esc(h_["page"])}</b>'
                  f'<span style="color:#5B7078;font-size:.8rem">similarity {h_["score"]}</span></div><div class="bar"><i style="width:{pct}%"></i></div>{body}</div>')

# =========================== ANALYSIS ===========================
elif page == "Analysis":
    if not docs:
        empty("Nothing to analyse yet. Add a filing in Documents.")
    else:
        pick_d = st.selectbox("Company", docs, format_func=doc_label)
        flags = db.get_flags(pick_d["id"])
        lvl, col = risk_level(flags)
        H(f'<div style="margin:6px 0 12px">Risk indicator (from flag counts): <span class="b {col}">{lvl}</span></div>')
        latest = comparison_agent.latest_metrics(pick_d["id"])
        show = [n for n in ("Revenue", "Net Profit", "Net Margin %", "Debt-to-Equity") if n in latest]
        for c, n in zip(st.columns(max(len(show), 1)), show):
            m = latest[n]
            c.metric(n, f"{m['value_num']:,.2f}", help=f"{m['unit']} · {m['period']} · page {m['page']}")
        st.markdown(f"#### 🚩 Red flags ({len(flags)})")
        sev = st.radio("Show", ["all", "high", "medium", "low"], horizontal=True, label_visibility="collapsed")
        shown = [f for f in flags if sev == "all" or f["severity"] == sev]
        if not shown:
            st.success("No red flags in this view.")
        H("".join(flag_html(f) for f in shown))
        mets = db.get_metrics(pick_d["id"])
        with st.expander(f"📈 All extracted metrics ({len(mets)})"):
            if mets:
                df = pd.DataFrame(mets)
                df["status"] = df.apply(lambda r: "computed" if r["source"] == "computed" else ("✅ verified" if r["verified"] else "⚠️ unverified"), axis=1)
                st.dataframe(df[["name", "period", "value", "unit", "page", "status", "quote"]].sort_values(["name", "period"]),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No metrics extracted yet.")

# =========================== COMPARE ===========================
elif page == "Compare":
    comparison_docs = db.analyzed_documents()
    if len(comparison_docs) < 2:
        empty("Comparison needs at least two analyzed PDFs. Analyze more filings in any workspace.")
    else:
        st.caption("Choose analyzed PDFs from any workspace. The two newest analyzed PDFs are selected by default.")
        doc_options = [d["id"] for d in comparison_docs]
        doc_by_id = {d["id"]: d for d in comparison_docs}
        chosen = st.multiselect("Analyzed PDFs", doc_options, default=doc_options[:2],
                                format_func=lambda i: f"{doc_by_id[i]['company']} - {doc_by_id[i]['filename']}")
        if len(chosen) >= 2:
            cdf, units, win = comparison_agent.build_table(chosen)
            disp = cdf.copy().astype(object)
            for n_, row in cdf.iterrows():
                for c in cdf.columns:
                    disp.loc[n_, c] = "n/a" if pd.isna(row[c]) else f"{row[c]:,.2f} {units.get(n_, {}).get(c, '')}"
            st.dataframe(disp, use_container_width=True)
            if win:
                st.markdown("**🏆 Ratio leaders:** " + "  ·  ".join(f"{k} → **{v}**" for k, v in win.items()))
            pc = [m for m in ("Net Margin %", "Operating Margin %", "Return on Equity %") if m in cdf.index]
            if pc:
                st.markdown("#### Profitability comparison (%)")
                st.bar_chart(cdf.loc[pc])
            comp_key = f"comp_{sid}_{'_'.join(sorted(chosen))}"
            if st.button("✍️ Generate peer commentary"):
                with st.spinner("Comparison Agent is writing..."):
                    st.session_state[comp_key] = comparison_agent.commentary(chosen)
            if st.session_state.get(comp_key):
                st.info(md(st.session_state[comp_key]))

# =========================== RESEARCH CHAT ===========================
elif page == "Research chat":
    if not docs:
        empty("Ingest a filing first, then ask questions about it here.")
    else:
        st.caption("Ask in plain English. Answers show their reasoning plan and cite the company and page.")
        if not messages:
            st.markdown("**Try one of these:**")
            sc = st.columns(2)
            for i, s_ in enumerate(SUGG):
                if sc[i % 2].button(s_, key=f"sg{i}", use_container_width=True):
                    st.session_state["pending_q"] = s_
                    st.rerun()
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(md(m["content"]))
                meta = m.get("meta")
                if meta and meta.get("plan"):
                    with st.expander("🧠 Reasoning plan"):
                        for i, p in enumerate(meta["plan"], 1):
                            st.markdown(md(f"{i}. {p}"))
                if meta and meta.get("sources"):
                    with st.expander(f"📚 Sources ({len(meta['sources'])})"):
                        for s_ in meta["sources"]:
                            st.markdown(f"**{s_['label']}**  ·  `{s_['filename']}`")
                            st.caption(md(s_["snippet"]))
                if meta and not meta.get("grounded", True):
                    st.warning("This answer has no source citations. Treat it as unverified.")
        q = st.chat_input("Ask about revenue, risks, margins, comparisons...") or st.session_state.pop("pending_q", None)
        if q:
            db.add_message(sid, "user", q)
            with st.spinner("Research Agent: planning, retrieving, reasoning..."):
                try:
                    res = research_agent.answer(q, doc_ids, history=messages)
                    db.add_message(sid, "assistant", res["answer"], {"plan": res["plan"], "sources": res["sources"], "grounded": res["grounded"]})
                except Exception as e:
                    db.add_message(sid, "assistant", f"Sorry, something went wrong: {e}")
            st.rerun()

# =========================== REPORT ===========================
elif page == "Report":
    if not analysed:
        empty("Your report appears here once at least one filing is analysed.")
    else:
        st.caption("PDF sections: executive summary, key financials, red flags, comparison, outlook and methodology.")
        chosen = st.multiselect("Companies to include", [d["id"] for d in analysed], default=[d["id"] for d in analysed], key="rep_docs",
                                format_func=lambda i: next(d["company"] for d in analysed if d["id"] == i))
        if chosen and st.button("📑 Generate PDF report", type="primary"):
            out = config.REPORT_DIR / f"report_{sid}_{int(time.time())}.pdf"
            with st.status("Report Agent is compiling...", expanded=True) as s_:
                report_agent.generate_report(current["name"], chosen, out, progress=st.write)
                s_.update(label="Report ready", state="complete")
            st.session_state[f"report_{sid}"] = str(out)
        path = st.session_state.get(f"report_{sid}")
        if path:
            with open(path, "rb") as fh:
                st.download_button("⬇️ Download PDF report", fh.read(), file_name=f"{current['name']}_report.pdf", mime="application/pdf", type="primary")

H('<div class="foot">FinResearch AI · Multi-Agent Financial Research · Infosys Springboard Internship</div>')
