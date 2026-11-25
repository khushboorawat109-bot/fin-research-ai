"""Multi-agent orchestration layer.

On upload (sequential):  Document Agent -> Extraction Agent -> Red Flag Agent
On demand:               Research Agent, Comparison Agent, Report Agent
Each stage reports progress to the UI; a failure in one stage never loses earlier results.
"""
import db
from agents import document_agent, extraction_agent, red_flag_agent


def analyze(doc_id: str, progress=None):
    say = progress or (lambda m: None)
    try:
        extraction_agent.run(doc_id, say)
        red_flag_agent.run(doc_id, say)
        db.set_doc_status(doc_id, "analyzed")
        return True
    except Exception as e:
        say(f"Analysis failed: {e}")
        db.set_doc_status(doc_id, "indexed")
        return False


def process_upload(path, session_id, progress=None):
    say = progress or (lambda m: None)
    doc = document_agent.ingest(path, progress=say)
    db.attach_document(session_id, doc["id"])
    if doc["status"] != "analyzed":
        analyze(doc["id"], say)
    return db.get_document(doc["id"])


def bootstrap(progress=None):
    """App start-up: init DB, index seed docs, analyse any document not analysed yet."""
    say = progress or (lambda m: None)
    db.init_db()
    document_agent.load_seed_documents(progress=say)
    for d in db.unanalyzed_documents():
        say(f'Analysing {d["company"]}...')
        analyze(d["id"], say)
