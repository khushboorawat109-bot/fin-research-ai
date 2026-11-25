"""Offline tests: no API key, no network. Run:  python -m unittest tests.test_offline -v"""
import sys, tempfile, unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
config.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
import db, llm
from agents import (comparison_agent, document_agent, extraction_agent as ex,
                    red_flag_agent as rf, report_agent, research_agent)


def M(name, period, num, unit="INR crore", page=10, verified=1, source="extracted"):
    return dict(name=name, value=f"{num:,}", unit=unit, period=period, page=page, quote=f"{name} {num:,}",
                value_num=num, verified=verified, source=source)


class TestHelpers(unittest.TestCase):
    def test_chunking_keeps_pages(self):
        ch = document_agent.chunk_pages([(1, "Revenue grew. " * 200), (2, "tiny"), (3, "Debt rose.\n" * 150)])
        self.assertEqual({c["page"] for c in ch}, {1, 3})
        self.assertTrue(all(len(c["text"]) <= config.CHUNK_SIZE + 5 for c in ch))

    def test_query_embeddings_are_reused(self):
        calls = []

        class FakeModels:
            def embed_content(self, model, contents, config):
                calls.append(contents)
                return SimpleNamespace(embeddings=[
                    SimpleNamespace(values=[float(i)]) for i, _ in enumerate(contents)])

        old_client = document_agent._client
        document_agent._query_embedding_cache.clear()
        document_agent._client = lambda: SimpleNamespace(models=FakeModels())
        try:
            first = document_agent.embed_texts(["Revenue", "Revenue"], task_type="RETRIEVAL_QUERY")
            second = document_agent.embed_texts(["Revenue"], task_type="RETRIEVAL_QUERY")
        finally:
            document_agent._client = old_client
            document_agent._query_embedding_cache.clear()
        self.assertEqual(first, [[0.0], [0.0]])
        self.assertEqual(second, [[0.0]])
        self.assertEqual(calls, [["Revenue"]])

    def test_period_normalisation(self):
        for raw, exp in [("FY24", "FY2024"), ("2023-24", "FY2024"), ("Year ended March 31, 2024", "FY2024"),
                         ("fy2023", "FY2023"), ("2022-2023", "FY2023")]:
            self.assertEqual(ex.norm_period(raw), exp, raw)

    def test_verify(self):
        chunk = "Revenue from operations   1,53,670  crore\nfor the year"
        ok = {"quote": "Revenue from operations 1,53,670 crore", "value": "1,53,670"}
        bad_quote = {"quote": "Revenue was 9,99,999", "value": "9,99,999"}
        bad_num = {"quote": "Revenue from operations 1,53,670 crore", "value": "1,60,000"}
        self.assertTrue(ex.verify(ok, chunk))
        self.assertFalse(ex.verify(bad_quote, chunk))
        self.assertFalse(ex.verify(bad_num, chunk))

    def test_json_parser(self):
        self.assertEqual(llm.parse_json('```json\n[{"a":1}]\n```'), [{"a": 1}])
        self.assertEqual(llm.parse_json('Here: {"x": 2} done'), {"x": 2})
        self.assertIsNone(llm.parse_json("nope"))

    def test_daily_quota_error_fails_without_retry(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("429 RESOURCE_EXHAUSTED: requests per day limit exceeded")

        old_last_call = llm._last_call
        llm._last_call = 0
        try:
            with patch.object(llm, "client", return_value=SimpleNamespace(models=FakeModels())):
                with patch.object(llm.time, "sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, "daily request quota"):
                        llm.generate("test")
                    sleep.assert_not_called()
        finally:
            llm._last_call = old_last_call


class TestExtraction(unittest.TestCase):
    def setUp(self):
        db.init_db()
        self.hits = [{"id": "a", "text": "Revenue from operations 153,670 crore (FY2024) vs 146,767 (FY2023). "
                                          "Net profit 26,248 crore.", "page": 57, "company": "Acme",
                      "filename": "acme.pdf", "doc_id": "d1", "score": .9}]
        document_agent.retrieve_many = lambda *a, **k: self.hits
        self.items = [
            {"metric": "Revenue", "period": "2023-24", "value": "153,670", "value_num": 153670, "unit": "INR crore",
             "chunk": "C1", "quote": "Revenue from operations 153,670 crore"},
            {"metric": "Revenue", "period": "FY23", "value": "146,767", "value_num": 146767, "unit": "INR crore",
             "chunk": "C1", "quote": "vs 146,767"},
            {"metric": "Net Profit", "period": "FY2024", "value": "26,248", "value_num": 26248, "unit": "INR crore",
             "chunk": "C1", "quote": "Net profit 26,248 crore"},
            {"metric": "EPS", "period": "FY2024", "value": "63.0", "value_num": 63.0, "unit": "INR",
             "chunk": "C1", "quote": "EPS was 63.0"},                    # hallucinated quote
            {"metric": "Nonsense", "period": "FY2024", "value": "1", "chunk": "C1", "quote": "x"},
        ]
        llm.generate_json = lambda *a, **k: self.items

    def test_extract_and_verify(self):
        rows = ex.extract_group("d1", "Income statement", ex.METRIC_GROUPS["Income statement"])
        names = {(r["name"], r["period"]): r["verified"] for r in rows}
        self.assertEqual(names[("Revenue", "FY2024")], 1)
        self.assertEqual(names[("Revenue", "FY2023")], 1)
        self.assertEqual(names[("EPS", "FY2024")], 0)       # unverified quote
        self.assertNotIn(("Nonsense", "FY2024"), names)       # unknown metric dropped

    def test_run_extracts_all_groups_with_one_llm_request(self):
        db.init_db()
        calls = []

        def generate_json(prompt, *args, **kwargs):
            calls.append(prompt)
            return self.items

        llm.generate_json = generate_json
        ex.run("d-run")
        self.assertEqual(len(calls), 1)
        self.assertIn("Employee Count", calls[0])

    def test_run_propagates_model_failure(self):
        db.init_db()
        with patch.object(llm, "generate_json", side_effect=RuntimeError("quota exceeded")):
            with self.assertRaisesRegex(RuntimeError, "quota exceeded"):
                ex.run("d-failed")
        self.assertEqual(db.get_metrics("d-failed"), [])

    def test_ratios(self):
        rows = [M("Revenue", "FY2024", 1000), M("Net Profit", "FY2024", 200), M("Total Debt", "FY2024", 300),
                M("Total Equity", "FY2024", 600), M("Current Assets", "FY2024", 500), M("Current Liabilities", "FY2024", 250)]
        got = {r["name"]: r["value_num"] for r in ex.compute_ratios(rows)}
        self.assertEqual(got["Net Margin %"], 20.0)
        self.assertEqual(got["Debt-to-Equity"], 0.5)
        self.assertEqual(got["Current Ratio"], 2.0)

    def test_unverified_not_used_in_ratios(self):
        rows = [M("Revenue", "FY2024", 1000), M("Net Profit", "FY2024", 200, verified=0)]
        self.assertEqual(ex.compute_ratios(rows), [])


class TestRedFlags(unittest.TestCase):
    def cats(self, rows):
        return {f["category"]: f["severity"] for f in rf.rule_flags(rows)}

    def test_trouble_company(self):
        rows = [M("Revenue", "FY2024", 800), M("Revenue", "FY2023", 1000),
                M("Net Profit", "FY2024", 50), M("Net Profit", "FY2023", 120),
                M("Total Debt", "FY2024", 900), M("Total Debt", "FY2023", 500),
                M("Debt-to-Equity", "FY2024", 3.5, unit="x", source="computed"),
                M("Current Ratio", "FY2024", 0.6, unit="x", source="computed"),
                M("Net Margin %", "FY2024", 6.25, "%", source="computed"), M("Net Margin %", "FY2023", 12, "%", source="computed"),
                M("Operating Cash Flow", "FY2024", -10)]
        c = self.cats(rows)
        for k in ("Falling revenue", "Falling profit", "Rising debt", "High leverage", "Liquidity risk",
                  "Margin compression", "Negative operating cash flow"):
            self.assertIn(k, c, k)
        self.assertEqual(c["High leverage"], "high"); self.assertEqual(c["Falling profit"], "high")

    def test_healthy_company_clean(self):
        rows = [M("Revenue", "FY2024", 1100), M("Revenue", "FY2023", 1000), M("Net Profit", "FY2024", 220),
                M("Net Profit", "FY2023", 200), M("Current Ratio", "FY2024", 2.1, "x", source="computed"),
                M("Operating Cash Flow", "FY2024", 250)]
        self.assertEqual(self.cats(rows), {})

    def test_text_flags_require_verified_quote(self):
        hit = {"id": "a", "text": "The auditor has issued a qualified opinion on receivables recoverability.",
               "page": 120, "company": "X", "filename": "x.pdf", "doc_id": "d", "score": .8}
        document_agent.retrieve_many = lambda *a, **k: [hit]
        llm.generate_json = lambda *a, **k: [
            {"category": "Auditor qualification", "severity": "high", "description": "Qualified opinion.",
             "chunk": "C1", "quote": "issued a qualified opinion on receivables"},
            {"category": "Going concern", "severity": "high", "description": "Doubt.", "chunk": "C1",
             "quote": "substantial doubt about going concern"}]
        out = rf.text_flags("d")
        self.assertEqual([f["category"] for f in out], ["Auditor qualification"])


class TestResearch(unittest.TestCase):
    def test_citation_mapping(self):
        src = {"S1": {"company": "Acme", "page": 5, "filename": "a.pdf", "text": "t", "score": .9},
               "S2": {"company": "Beta", "page": 9, "filename": "b.pdf", "text": "u", "score": .8}}
        r = research_agent.finalize("Step 1: Acme revenue rose [S1]. Beta lower [S2][S7]. **Answer**: Acme.", src, ["q"])
        self.assertIn("[Acme, p.5]", r["answer"]); self.assertIn("[Beta, p.9]", r["answer"])
        self.assertEqual(r["invalid_citations"], ["S7"]); self.assertEqual(len(r["sources"]), 2)
        self.assertTrue(r["grounded"]); self.assertNotIn("**", r["answer"])

    def test_ungrounded_detected(self):
        r = research_agent.finalize("Revenue is huge.", {}, ["q"])
        self.assertFalse(r["grounded"])
        r = research_agent.finalize("Not found in the provided documents.", {}, ["q"])
        self.assertTrue(r["grounded"])


class TestComparisonAndReport(unittest.TestCase):
    def setUp(self):
        db.init_db()
        for i, (co, rev, np_, debt) in enumerate([("Alpha Ltd", 1000, 200, 100), ("Beta Ltd", 900, 90, 400)]):
            did = f"doc{i}"
            if not db.get_document(did):
                db.add_document(dict(id=did, filename=f"{co}.pdf", company=co, file_hash=did, n_pages=100,
                                     n_chunks=50, is_seed=0, status="indexed"))
            rows = [M("Revenue", "FY2024", rev), M("Net Profit", "FY2024", np_), M("Total Debt", "FY2024", debt),
                    M("Total Equity", "FY2024", 500), M("Revenue", "FY2023", rev - 50), M("Net Profit", "FY2023", np_ + 30)]
            db.replace_metrics(did, rows + ex.compute_ratios(rows))
            db.replace_flags(did, rf.rule_flags(db.get_metrics(did)) + [dict(category="Auditor qualification",
                             severity="high", description="Qualified opinion & <risk>.", page=3,
                             quote="qualified ₹ opinion", verified=1, source="llm")])
        self.ids = ["doc0", "doc1"]

    def test_analyzed_documents_can_be_selected_across_workspaces(self):
        ids = ["global-doc-a", "global-doc-b"]
        for i, did in enumerate(ids):
            db.add_document(dict(id=did, filename=f"Global{i}.pdf", company=f"Global{i}", file_hash=did,
                                 n_pages=10, n_chunks=5, is_seed=0, status="analyzed"))
        first, second = db.create_session("Global A"), db.create_session("Global B")
        db.attach_document(first, ids[0])
        db.attach_document(second, ids[1])

        available = {d["id"] for d in db.analyzed_documents()}
        self.assertTrue(set(ids).issubset(available))
        self.assertNotIn(ids[1], {d["id"] for d in db.session_documents(first)})
        self.assertNotIn(ids[0], {d["id"] for d in db.session_documents(second)})

    def test_delete_session_removes_workspace_and_chat_but_keeps_pdf(self):
        sid = db.create_session("Delete test")
        did = "delete-keeps-doc"
        db.add_document(dict(id=did, filename="Keep.pdf", company="Keep", file_hash=did,
                             n_pages=10, n_chunks=5, is_seed=0, status="analyzed"))
        db.attach_document(sid, did)
        db.add_message(sid, "user", "test message")

        db.delete_session(sid)

        self.assertNotIn(sid, {session["id"] for session in db.list_sessions()})
        self.assertEqual(db.get_messages(sid), [])
        self.assertIsNotNone(db.get_document(did))
        self.assertIn(did, {doc["id"] for doc in db.analyzed_documents()})

    def test_comparison(self):
        df, units, winners = comparison_agent.build_table(self.ids)
        self.assertEqual(list(df.columns), ["Alpha Ltd", "Beta Ltd"])
        self.assertEqual(winners["Net Margin %"], "Alpha Ltd"); self.assertEqual(winners["Debt-to-Equity"], "Alpha Ltd")

    def test_pdf(self):
        out = Path(tempfile.mkdtemp()) / "r.pdf"
        hit = {"id": "z", "text": "We expect growth.", "page": 8, "company": "x",
               "filename": "x", "doc_id": "d", "score": .5}
        with patch.object(llm, "generate", return_value="Plain summary with ₹ and (p.10)."), \
                patch.object(document_agent, "retrieve_many", return_value=[hit]):
            report_agent.generate_report("Test <Session>", self.ids, out)
        self.assertTrue(out.exists() and out.stat().st_size > 5000)
        self.assertEqual(out.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
