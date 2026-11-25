"""SQLite layer: sessions, documents, extracted metrics, red flags, chat history."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, filename TEXT NOT NULL, company TEXT,
    file_hash TEXT UNIQUE, n_pages INTEGER, n_chunks INTEGER,
    is_seed INTEGER DEFAULT 0, status TEXT DEFAULT 'indexed',
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS session_documents (
    session_id TEXT, doc_id TEXT, PRIMARY KEY (session_id, doc_id));
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, name TEXT, value TEXT,
    unit TEXT, period TEXT, page INTEGER, quote TEXT,
    value_num REAL, verified INTEGER DEFAULT 0, source TEXT DEFAULT 'extracted');
CREATE TABLE IF NOT EXISTS red_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, category TEXT,
    severity TEXT, description TEXT, page INTEGER, quote TEXT,
    verified INTEGER DEFAULT 0, source TEXT DEFAULT 'llm');
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
    content TEXT, created_at TEXT, meta TEXT);
"""

# columns added after Milestone 1 (so an old app.db keeps working)
MIGRATIONS = {
    "metrics": {"value_num": "REAL", "verified": "INTEGER DEFAULT 0",
                "source": "TEXT DEFAULT 'extracted'"},
    "red_flags": {"verified": "INTEGER DEFAULT 0", "source": "TEXT DEFAULT 'llm'"},
    "chat_messages": {"meta": "TEXT"},
}


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols.items():
                if col not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------- sessions ----------------
def create_session(name: str) -> str:
    sid = uuid.uuid4().hex[:8]
    with conn() as c:
        c.execute("INSERT INTO sessions VALUES (?,?,?)", (sid, name.strip(), _now()))
        c.execute("""INSERT OR IGNORE INTO session_documents
                     SELECT ?, id FROM documents WHERE is_seed=1""", (sid,))
    return sid


def list_sessions():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM sessions ORDER BY created_at DESC")]


def delete_session(sid: str):
    with conn() as c:
        c.execute("DELETE FROM session_documents WHERE session_id=?", (sid,))
        c.execute("DELETE FROM chat_messages WHERE session_id=?", (sid,))
        c.execute("DELETE FROM sessions WHERE id=?", (sid,))


# ---------------- documents ----------------
def get_doc_by_hash(h: str):
    with conn() as c:
        r = c.execute("SELECT * FROM documents WHERE file_hash=?", (h,)).fetchone()
        return dict(r) if r else None


def get_document(doc_id: str):
    with conn() as c:
        r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None


def add_document(doc: dict):
    with conn() as c:
        c.execute("""INSERT INTO documents
            (id, filename, company, file_hash, n_pages, n_chunks, is_seed, status, created_at)
            VALUES (:id,:filename,:company,:file_hash,:n_pages,:n_chunks,:is_seed,:status,:created_at)""",
                  {**doc, "created_at": _now()})
        if doc["is_seed"]:
            c.execute("INSERT OR IGNORE INTO session_documents SELECT id, ? FROM sessions", (doc["id"],))


def set_doc_status(doc_id: str, status: str):
    with conn() as c:
        c.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))


def attach_document(session_id: str, doc_id: str):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO session_documents VALUES (?,?)", (session_id, doc_id))


def detach_document(session_id: str, doc_id: str):
    with conn() as c:
        c.execute("DELETE FROM session_documents WHERE session_id=? AND doc_id=?", (session_id, doc_id))


def session_documents(session_id: str):
    with conn() as c:
        rows = c.execute("""SELECT d.* FROM documents d
            JOIN session_documents sd ON sd.doc_id=d.id
            WHERE sd.session_id=? ORDER BY d.created_at""", (session_id,))
        return [dict(r) for r in rows]


def analyzed_documents():
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM documents WHERE status='analyzed' ORDER BY created_at DESC, id")
        return [dict(r) for r in rows]


def unanalyzed_documents():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM documents WHERE status!='analyzed'")]


# ---------------- metrics / flags ----------------
def replace_metrics(doc_id: str, rows: list):
    with conn() as c:
        c.execute("DELETE FROM metrics WHERE doc_id=?", (doc_id,))
        c.executemany("""INSERT INTO metrics
            (doc_id,name,value,unit,period,page,quote,value_num,verified,source)
            VALUES (:doc_id,:name,:value,:unit,:period,:page,:quote,:value_num,:verified,:source)""",
                      [{**r, "doc_id": doc_id} for r in rows])


def get_metrics(doc_id: str):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM metrics WHERE doc_id=? ORDER BY id", (doc_id,))]


def replace_flags(doc_id: str, rows: list):
    with conn() as c:
        c.execute("DELETE FROM red_flags WHERE doc_id=?", (doc_id,))
        c.executemany("""INSERT INTO red_flags
            (doc_id,category,severity,description,page,quote,verified,source)
            VALUES (:doc_id,:category,:severity,:description,:page,:quote,:verified,:source)""",
                      [{**r, "doc_id": doc_id} for r in rows])


def get_flags(doc_id: str):
    order = "CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END"
    with conn() as c:
        return [dict(r) for r in c.execute(
            f"SELECT * FROM red_flags WHERE doc_id=? ORDER BY {order}, id", (doc_id,))]


# ---------------- chat ----------------
def add_message(session_id: str, role: str, content: str, meta: dict | None = None):
    with conn() as c:
        c.execute("INSERT INTO chat_messages (session_id,role,content,created_at,meta) VALUES (?,?,?,?,?)",
                  (session_id, role, content, _now(), json.dumps(meta) if meta else None))


def get_messages(session_id: str):
    with conn() as c:
        rows = c.execute("SELECT * FROM chat_messages WHERE session_id=? ORDER BY id", (session_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"]) if d["meta"] else None
            out.append(d)
        return out
