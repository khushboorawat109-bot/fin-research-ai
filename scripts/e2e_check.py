"""End-to-end validation on the seed documents (needs GEMINI_API_KEY + PDFs in data/seed).

    python scripts/e2e_check.py

Checks agent collaboration, extraction coverage/faithfulness, red-flag output, citation
grounding, comparison and PDF completeness. If tests/golden.json exists it also reports
extraction ACCURACY against your hand-entered ground truth.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config, db, orchestrator
from agents import comparison_agent, report_agent, research_agent

CORE = ["Revenue", "Net Profit", "Total Assets", "Total Equity"]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {name} {detail}')


orchestrator.bootstrap(progress=print)
with db.conn() as c:
    docs = [dict(r) for r in c.execute("SELECT * FROM documents WHERE is_seed=1")]
check("seed documents loaded (3-4 expected)", len(docs) >= 3, f"({len(docs)})")

golden = json.loads(Path("tests/golden.json").read_text()) if Path("tests/golden.json").exists() else {}
hit = tot = 0
for d in docs:
    m = db.get_metrics(d["id"])
    ver = [x for x in m if x["verified"] and x["source"] == "extracted"]
    names = {x["name"] for x in ver}
    check(f'{d["company"]}: analysed', d["status"] == "analyzed")
    check(f'{d["company"]}: core metrics present', all(c in names for c in CORE), f"(missing {[c for c in CORE if c not in names]})")
    rate = len(ver) / max(1, len([x for x in m if x["source"] == "extracted"]))
    check(f'{d["company"]}: verification rate >= 80%', rate >= .8, f"({rate:.0%})")
    print(f'        red flags: {len(db.get_flags(d["id"]))}')
    for period, truth in golden.get(d["company"], {}).items():
        for name, val in truth.items():
            if not val:
                continue
            got = next((x["value_num"] for x in m if x["name"] == name and x["period"] == period), None)
            tot += 1
            ok = got is not None and abs(got - val) <= abs(val) * 0.005
            hit += ok
            print(f'        golden {name} {period}: expected {val}, got {got} -> {"OK" if ok else "MISS"}')
if tot:
    print(f"\nEXTRACTION ACCURACY vs ground truth: {hit}/{tot} = {hit / tot:.0%}")

ids = [d["id"] for d in docs]
q = "Compare the net profit and operating margin of the companies and explain which is healthier."
r = research_agent.answer(q, ids)
check("research answer is grounded with citations", r["grounded"] and len(r["sources"]) > 0)
check("no invalid citations", not r["invalid_citations"], str(r["invalid_citations"]))
print("\n--- sample answer ---\n", r["answer"][:900], "\n")
if len(ids) >= 2:
    df, _, _ = comparison_agent.build_table(ids)
    check("comparison table built", df.shape[1] >= 2 and not df.empty)
out = config.REPORT_DIR / "e2e_report.pdf"
report_agent.generate_report("E2E check", ids, out)
check("PDF report generated", out.exists() and out.stat().st_size > 10_000, str(out))
print(f"\n{sum(results)}/{len(results)} checks passed")
