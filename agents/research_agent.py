"""Research Agent: multi-step retrieval + reasoning with strict source citations."""
import re

import llm
from agents import document_agent

PLAN_SYSTEM = "You break financial research questions into search sub-questions. Return JSON only."
PLAN_PROMPT = """Conversation so far:
{history}

New question: {q}

Break it into 1-4 short, self-contained sub-questions that can each be answered by searching a company
filing (mention the company or metric explicitly; resolve pronouns using the conversation).
Return a JSON array of strings."""

ANS_SYSTEM = """You are a financial research assistant. Answer using ONLY the numbered sources.
Rules:
- Work step by step: "Step 1: ...", "Step 2: ...", then "Answer: ...".
- Cite every factual claim with its source label like [S3]. Several labels are allowed: [S1][S4].
- Quote numbers exactly as in the sources and keep their units/currency.
- If the sources do not contain the information, say "Not found in the provided documents." for that part.
- Never use outside knowledge. Do not invent figures. Plain text, no markdown tables."""

ANS_PROMPT = """Question: {q}

Sub-questions investigated: {plan}

SOURCES:
{sources}
"""


def _history_text(history, n=4):
    h = (history or [])[-n:]
    return "\n".join(f'{m["role"]}: {m["content"][:400]}' for m in h) or "(none)"


def answer(question: str, doc_ids: list, history: list | None = None):
    if not doc_ids:
        return {"answer": "No documents in this session. Add a document first.", "plan": [], "sources": [],
                "invalid_citations": [], "grounded": False}
    plan = llm.generate_json(PLAN_PROMPT.format(history=_history_text(history), q=question),
                             PLAN_SYSTEM, default=None)
    plan = [p for p in plan if isinstance(p, str)][:4] if isinstance(plan, list) and plan else [question]

    per_doc = len(doc_ids) <= 5
    hits = document_agent.retrieve_many(plan, doc_ids, k=3 if per_doc else 5, per_doc=per_doc)[:18]
    src = {f"S{i + 1}": h for i, h in enumerate(hits)}
    sources_txt = "\n\n".join(f'[{sid}] {h["company"]}, page {h["page"]}\n{h["text"]}' for sid, h in src.items())

    raw = llm.generate(ANS_PROMPT.format(q=question, plan="; ".join(plan), sources=sources_txt), ANS_SYSTEM)
    return finalize(raw, src, plan)


def finalize(raw: str, src: dict, plan: list):
    """Validate citations and convert [S#] -> [Company, p.N]. Pure function (tested offline)."""
    used, invalid = [], []

    def repl(m):
        sid = m.group(1)
        if sid in src:
            if sid not in used:
                used.append(sid)
            return f'[{src[sid]["company"]}, p.{src[sid]["page"]}]'
        invalid.append(sid)
        return ""

    text = re.sub(r"\[(S\d+)\]", repl, raw)
    text = text.replace("**", "").strip()
    not_found = "not found in the provided documents" in text.lower()
    grounded = bool(used) or not_found
    sources = [{"label": f'{src[s]["company"]}, p.{src[s]["page"]}', "filename": src[s]["filename"],
                "snippet": src[s]["text"][:500], "score": src[s]["score"]} for s in used]
    return {"answer": text, "plan": plan, "sources": sources, "invalid_citations": invalid, "grounded": grounded}
