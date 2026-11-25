"""Document Agent: upload -> parse -> chunk -> embed -> index in ChromaDB.

Every chunk keeps (doc_id, company, filename, page) so later agents can cite
exact sources, e.g. [Infosys AR FY24, p.57].
"""
import hashlib
import re
import time
import uuid
from pathlib import Path

import config
import db

_chroma_collection = None
_genai_client = None
_query_embedding_cache = {}


# ---------- external clients (lazy so the module imports without them) ----------
def _collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            "filings", metadata={"hnsw:space": "cosine"})
    return _chroma_collection


def _client():
    import llm
    return llm.client()


# ---------- parsing ----------
def parse_document(path: Path):
    """Return list of (page_number, text). Supports PDF and TXT."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF
        pages = []
        with fitz.open(path) as pdf:
            for i, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((i, text))
        return pages
    if path.suffix.lower() == ".txt":
        return [(1, path.read_text(errors="ignore"))]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def guess_company(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"(?i)(annual[_\- ]?report|10-?k|fy\d{2,4}|\d{4}|[_\-]+)", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().title() or "Unknown"


# ---------- chunking ----------
def chunk_pages(pages, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Page-aware chunking: chunks never cross pages, so every chunk has one page number."""
    chunks = []
    for page_no, raw in pages:
        text = clean_text(raw)
        if len(text) < 40:
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):  # try to break at a sentence/line boundary
                cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if cut > start + size // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


# ---------- embeddings ----------
def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    from google.genai import types
    if task_type == "RETRIEVAL_QUERY":
        keys = [(config.EMBED_MODEL, text) for text in texts]
        vectors = {key: _query_embedding_cache[key] for key in keys
                   if key in _query_embedding_cache}
        missing = list(dict.fromkeys(key for key in keys if key not in _query_embedding_cache))
        for i in range(0, len(missing), config.EMBED_BATCH):
            batch_keys = missing[i:i + config.EMBED_BATCH]
            batch = [text for _, text in batch_keys]
            for attempt in range(5):  # backoff for free-tier rate limits
                try:
                    res = _client().models.embed_content(
                        model=config.EMBED_MODEL, contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type))
                    for key, embedding in zip(batch_keys, res.embeddings):
                        vectors[key] = embedding.values
                        _query_embedding_cache[key] = embedding.values
                        if len(_query_embedding_cache) > 256:
                            _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt * 5)
        return [vectors[key] for key in keys]

    out = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i:i + config.EMBED_BATCH]
        for attempt in range(5):  # backoff for free-tier rate limits
            try:
                res = _client().models.embed_content(
                    model=config.EMBED_MODEL, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type))
                out.extend(e.values for e in res.embeddings)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt * 5)
    return out


# ---------- agent API ----------
def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ingest(path, is_seed=False, company=None, progress=None):
    """Full pipeline for one file. Returns the document row (existing one if duplicate)."""
    path = Path(path)
    say = progress or (lambda msg: None)
    h = file_hash(path)
    existing = db.get_doc_by_hash(h)
    if existing:
        say(f"{path.name}: already indexed, reusing")
        return existing

    say(f"{path.name}: parsing...")
    pages = parse_document(path)
    if not pages:
        raise ValueError("No extractable text (scanned PDF?). OCR is not supported yet.")

    chunks = chunk_pages(pages)
    say(f"{path.name}: {len(pages)} pages -> {len(chunks)} chunks, embedding...")
    vectors = embed_texts([c["text"] for c in chunks])

    doc_id = uuid.uuid4().hex[:10]
    company = company or guess_company(path.name)
    _collection().add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": doc_id, "company": company,
                    "filename": path.name, "page": c["page"]} for c in chunks],
    )
    doc = {"id": doc_id, "filename": path.name, "company": company, "file_hash": h,
           "n_pages": len(pages), "n_chunks": len(chunks),
           "is_seed": int(is_seed), "status": "indexed"}
    db.add_document(doc)
    say(f"{path.name}: indexed")
    return doc


def retrieve_many(queries, doc_ids, k=4, per_doc=False):
    """Embed all queries in ONE API call, then search Chroma locally.
    per_doc=True guarantees every document is represented (needed for comparisons).
    Returns de-duplicated hits sorted by similarity."""
    if not doc_ids or not queries:
        return []
    qvs = embed_texts(list(queries), task_type="RETRIEVAL_QUERY")
    col = _collection()
    filters = [{"doc_id": d} for d in doc_ids] if per_doc else [{"doc_id": {"$in": list(doc_ids)}}]
    best = {}
    for qv in qvs:
        for flt in filters:
            res = col.query(query_embeddings=[qv], n_results=k, where=flt)
            for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                             res["metadatas"][0], res["distances"][0]):
                hit = {"id": cid, "text": text, "page": meta["page"], "company": meta["company"],
                       "filename": meta["filename"], "doc_id": meta["doc_id"],
                       "score": round(1 - dist, 3)}
                if cid not in best or hit["score"] > best[cid]["score"]:
                    best[cid] = hit
    return sorted(best.values(), key=lambda h: -h["score"])


def retrieve(query: str, doc_ids: list, k: int = 6):
    return retrieve_many([query], doc_ids, k=k)[:k]


def load_seed_documents(progress=None):
    """Index every PDF/TXT in data/seed/ once (hash-deduplicated)."""
    docs = []
    for f in sorted(config.SEED_DIR.iterdir()):
        if f.suffix.lower() in (".pdf", ".txt"):
            docs.append(ingest(f, is_seed=True, progress=progress))
    return docs
