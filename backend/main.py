from __future__ import annotations

"""ResearchPilot / 研途智研 API.

Drop-in FastAPI backend for the existing Next.js frontend. The implementation
keeps the original endpoint contracts while adding persistent storage,
chunked PDF processing, hybrid retrieval, traceable evidence and a safe tool
registry for the multi-agent orchestrator.
"""

import asyncio
import difflib
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import shutil
import sqlite3
import statistics
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional, Annotated, Literal
from urllib.parse import quote

import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # OCR is optional at startup; standard PDFs still work.
    RapidOCR = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("researchpilot")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
PAPER_DIR = DATA_DIR / "papers"
MEDIA_DIR = DATA_DIR / "media"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "researchpilot.sqlite3")).resolve()
for directory in (DATA_DIR, PAPER_DIR, MEDIA_DIR, DB_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "60"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "1200"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "7"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-2.0"))
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "72"))
FRONTEND_ORIGINS = [x.strip() for x in os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if x.strip()]

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None

# Optional dense retrieval. Configure an OpenAI-compatible embeddings endpoint,
# or install sentence-transformers and set LOCAL_EMBEDDING_MODEL.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "0").strip().lower() in ("1", "true", "yes", "on")
embedding_client = (
    OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    if EMBEDDING_API_KEY and EMBEDDING_MODEL else None
)
_local_embedding_model = None
_reranker_model = None
_ocr_engine = None

app = FastAPI(
    title="ResearchPilot 研途智研 API",
    version="2.0.0",
    description="Evidence-grounded multi-agent research assistant",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
)

security = HTTPBearer(auto_error=False)

# -----------------------------------------------------------------------------
# SQLite persistence
# -----------------------------------------------------------------------------
@contextmanager
def db() -> Iterable[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            full_text TEXT NOT NULL DEFAULT '',
            page_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(user_id, sha256)
        );
        CREATE INDEX IF NOT EXISTS idx_papers_user ON papers(user_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS pages (
            paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            page_no INTEGER NOT NULL,
            text TEXT NOT NULL,
            extraction_method TEXT NOT NULL DEFAULT 'text',
            quality REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY(paper_id, page_no)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            page_no INTEGER NOT NULL,
            section TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            embedding_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id, page_no);
        CREATE TABLE IF NOT EXISTS media_assets (
            filename TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_media_assets_user ON media_assets(user_id);
        """)


@app.on_event("startup")
def startup() -> None:
    init_db()
    log.info("ResearchPilot ready. LLM=%s, dense_retrieval=%s", bool(llm_client), bool(embedding_client or LOCAL_EMBEDDING_MODEL))


# -----------------------------------------------------------------------------
# Auth: PBKDF2 password hashing + random opaque bearer tokens stored by hash.
# -----------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _password_hash(password: str, salt: bytes) -> str:
    raw = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return raw.hex()


# Login protection: in-memory limits for single-instance deployments.
_LOGIN_RATE_LOCK = threading.Lock()
_LOGIN_REQUESTS: dict[str, deque[float]] = defaultdict(deque)
_LOGIN_FAILURES: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_LOGIN_BLOCKED_UNTIL: dict[tuple[str, str], float] = {}

_LOGIN_IP_WINDOW_SECONDS = 60
_LOGIN_IP_MAX_REQUESTS = 10
_LOGIN_IP_TRACKED_MAX = 10000
_LOGIN_FAILURE_WINDOW_SECONDS = 900
_LOGIN_FAILURE_MAX_ATTEMPTS = 5
_LOGIN_LOCK_SECONDS = 900


def _reset_login_rate_limits() -> None:
    """Reset login limiter state for isolated tests."""
    with _LOGIN_RATE_LOCK:
        _LOGIN_REQUESTS.clear()
        _LOGIN_FAILURES.clear()
        _LOGIN_BLOCKED_UNTIL.clear()


def _check_login_rate_limit(ip: str, username: str) -> None:
    now = time.monotonic()
    key = (ip, username.strip().casefold())

    with _LOGIN_RATE_LOCK:
        # Bound the number of tracked IPs without evicting active limits.
        if ip not in _LOGIN_REQUESTS and len(_LOGIN_REQUESTS) >= _LOGIN_IP_TRACKED_MAX:
            for old_ip in list(_LOGIN_REQUESTS):
                q = _LOGIN_REQUESTS[old_ip]
                while q and now - q[0] >= _LOGIN_IP_WINDOW_SECONDS:
                    q.popleft()
                if not q:
                    _LOGIN_REQUESTS.pop(old_ip, None)

            if len(_LOGIN_REQUESTS) >= _LOGIN_IP_TRACKED_MAX:
                raise HTTPException(
                    status_code=429,
                    detail="\u5f53\u524d\u767b\u5f55\u8bf7\u6c42\u8f83\u591a\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5",
                    headers={"Retry-After": str(_LOGIN_IP_WINDOW_SECONDS)},
                )

        requests = _LOGIN_REQUESTS[ip]
        while requests and now - requests[0] >= _LOGIN_IP_WINDOW_SECONDS:
            requests.popleft()

        if len(requests) >= _LOGIN_IP_MAX_REQUESTS:
            retry = max(1, int(_LOGIN_IP_WINDOW_SECONDS - (now - requests[0]) + 0.999))
            raise HTTPException(
                status_code=429,
                detail="\u767b\u5f55\u8bf7\u6c42\u8fc7\u4e8e\u9891\u7e41\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5",
                headers={"Retry-After": str(retry)},
            )

        requests.append(now)

        blocked_until = _LOGIN_BLOCKED_UNTIL.get(key)
        if blocked_until is not None:
            if now < blocked_until:
                retry = max(1, int(blocked_until - now + 0.999))
                raise HTTPException(
                    status_code=429,
                    detail="\u767b\u5f55\u5931\u8d25\u6b21\u6570\u8fc7\u591a\uff0c\u8bf7 15 \u5206\u949f\u540e\u91cd\u8bd5",
                    headers={"Retry-After": str(retry)},
                )
            _LOGIN_BLOCKED_UNTIL.pop(key, None)
            _LOGIN_FAILURES.pop(key, None)

        if len(_LOGIN_FAILURES) > 20000 or len(_LOGIN_BLOCKED_UNTIL) > 20000:
            for old_key in list(_LOGIN_FAILURES):
                q = _LOGIN_FAILURES[old_key]
                while q and now - q[0] >= _LOGIN_FAILURE_WINDOW_SECONDS:
                    q.popleft()
                if not q:
                    _LOGIN_FAILURES.pop(old_key, None)

            for old_key, blocked_until in list(_LOGIN_BLOCKED_UNTIL.items()):
                if now >= blocked_until:
                    _LOGIN_BLOCKED_UNTIL.pop(old_key, None)

            # Bound memory usage even under high-cardinality attack traffic.
            while len(_LOGIN_FAILURES) > 20000:
                _LOGIN_FAILURES.pop(next(iter(_LOGIN_FAILURES)))
            while len(_LOGIN_BLOCKED_UNTIL) > 20000:
                _LOGIN_BLOCKED_UNTIL.pop(next(iter(_LOGIN_BLOCKED_UNTIL)))


def _record_login_failure(ip: str, username: str) -> None:
    now = time.monotonic()
    key = (ip, username.strip().casefold())

    with _LOGIN_RATE_LOCK:
        failures = _LOGIN_FAILURES[key]
        while failures and now - failures[0] >= _LOGIN_FAILURE_WINDOW_SECONDS:
            failures.popleft()
        failures.append(now)

        if len(failures) >= _LOGIN_FAILURE_MAX_ATTEMPTS:
            _LOGIN_BLOCKED_UNTIL[key] = now + _LOGIN_LOCK_SECONDS


def _clear_login_failures(ip: str, username: str) -> None:
    key = (ip, username.strip().casefold())
    with _LOGIN_RATE_LOCK:
        _LOGIN_FAILURES.pop(key, None)
        _LOGIN_BLOCKED_UNTIL.pop(key, None)


_LOGIN_DUMMY_SALT = b"researchpilot-login-dummy-salt"
_LOGIN_DUMMY_HASH = _password_hash("not-a-real-password", _LOGIN_DUMMY_SALT)


def _issue_session(user_id: str) -> str:
    token = secrets.token_urlsafe(40)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = _now()
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions(token_hash,user_id,expires_at,created_at) VALUES(?,?,?,?)",
            (token_hash, user_id, _iso(now + timedelta(hours=TOKEN_TTL_HOURS)), _iso(now)),
        )
    return token


def current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="请先登录", headers={"WWW-Authenticate": "Bearer"})
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    with db() as conn:
        row = conn.execute(
            "SELECT u.id,u.username,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?",
            (token_hash,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        if datetime.fromisoformat(row["expires_at"]) < _now():
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return {"id": row["id"], "username": row["username"]}


@app.post("/api/auth/register")
def register(payload: dict):
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if len(username) < 2 or len(username) > 40:
        raise HTTPException(400, "用户名长度应为 2-40 个字符")
    if len(password) < 8:
        raise HTTPException(400, "密码至少需要 8 个字符")
    salt = secrets.token_bytes(16)
    user_id = uuid.uuid4().hex
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users(id,username,password_salt,password_hash,created_at) VALUES(?,?,?,?,?)",
                (user_id, username, salt.hex(), _password_hash(password, salt), _iso(_now())),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(409, "该用户名已被注册")
    return {"message": "注册成功，请登录"}


@app.post("/api/auth/login")
def login(payload: dict, request: Request):
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    ip = request.client.host if request.client else "unknown"

    _check_login_rate_limit(ip, username)

    with db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,),
        ).fetchone()

    if not row:
        hmac.compare_digest(
            _password_hash(password, _LOGIN_DUMMY_SALT),
            _LOGIN_DUMMY_HASH,
        )
        _record_login_failure(ip, username)
        raise HTTPException(401, "\u7528\u6237\u540d\u6216\u5bc6\u7801\u9519\u8bef")

    salt = bytes.fromhex(row["password_salt"])
    if not hmac.compare_digest(
        _password_hash(password, salt),
        row["password_hash"],
    ):
        _record_login_failure(ip, username)
        raise HTTPException(401, "\u7528\u6237\u540d\u6216\u5bc6\u7801\u9519\u8bef")

    _clear_login_failures(ip, username)
    token = _issue_session(row["id"])
    return {"message": "\u767b\u5f55\u6210\u529f", "token": token, "username": username}


@app.post("/api/auth/logout")
def logout(user: dict = Depends(current_user), credentials: HTTPAuthorizationCredentials = Depends(security)):
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()  # type: ignore[union-attr]
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
    return {"message": "已退出登录"}


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------
def llm_text(prompt: str, system: str = "你是一名严谨的科研助理。", temperature: float = 0.1) -> str:
    if not llm_client:
        raise HTTPException(503, "尚未配置大模型 API。请在 backend/.env 中设置 LLM_API_KEY。")
    _llm_started = time.perf_counter()
    result = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    content = result.choices[0].message.content or ""
    log.info(
        "[PERF][LLM] model=%s elapsed=%.2fs prompt_chars=%d output_chars=%d",
        LLM_MODEL,
        time.perf_counter() - _llm_started,
        len(prompt),
        len(content),
    )
    return content


def llm_json(prompt: str, system: str = "你是一个可靠的 JSON 输出器。") -> dict:
    raw = llm_text(prompt, system=system, temperature=0)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型没有返回合法 JSON")
    return json.loads(cleaned[start:end + 1])


# -----------------------------------------------------------------------------
# PDF parsing, metadata extraction and chunking
# -----------------------------------------------------------------------------
SECTION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,3}\s+)?([A-Z][A-Za-z0-9 ,:()\-/]{2,90})\s*$"
)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v(\d+))?", re.I)


def _normal_text(value: str) -> str:
    value = value.lower().replace("\u00ad", "")
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


ENGLISH_STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "what", "which", "who",
    "when", "where", "why", "how",
    "and", "or", "but",
    "of", "in", "on", "at",
    "to", "for", "from", "by",
    "with", "about", "as",
    "it", "its", "this", "that",
    "these", "those",
    "we", "they", "he", "she",
    "i", "you", "them", "their",
    "our", "your",
}


def _tokens(text: str) -> list[str]:
    # Original tokenizer retained for paper deduplication.
    tokens = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(list(segment))
        tokens.extend(segment[i:i + 2] for i in range(max(0, len(segment) - 1)))
    return tokens


def _retrieval_tokens(text: str) -> list[str]:
    # Stopword filtering applies only to BM25 retrieval.
    return [
        token for token in _tokens(text)
        if token not in ENGLISH_STOPWORDS
    ]


def _shingle_similarity(a: str, b: str, width: int = 5) -> float:
    aa, bb = _tokens(a[:120_000]), _tokens(b[:120_000])
    if not aa or not bb:
        return 0.0
    sa = set(tuple(aa[i:i + width]) for i in range(max(1, len(aa) - width + 1)))
    sb = set(tuple(bb[i:i + width]) for i in range(max(1, len(bb) - width + 1)))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None and RapidOCR is not None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _is_valid_heading(line: str) -> bool:
    if not line: return False
    line = line.strip()
    if len(line) > 65: return False  # 标题通常不会太长
    if re.search(r'[×=+±%\[\]]', line): return False  # 排除公式、引用等

    known_headings = {"abstract", "introduction", "related work", "method", "methods", "experiments", "results", "discussion", "conclusion", "future work", "references", "appendix"}
    if line.lower() in known_headings: return True

    # 匹配数字开头的章节
    m = re.match(r'^\d+(\.\d+)*\.?\s+([A-Za-z\u4e00-\u9fa5].*)', line)
    if m:
        rest = m.group(2)
        first_word = rest.split()[0]
        # 章节后面的第一个词必须是大写字母开头，或者是中文
        if not (first_word[0].isupper() or ('\u4e00' <= first_word[0] <= '\u9fa5')):
            return False
        # 排除包含量词/单位的行（如 "19 billion"）
        if any(w in line.lower() for w in ["billion", "million", "percent", "yards", "thousand", "sacks"]):
            return False
        # 排除包含 4 位数字的（如年份），除非是带小数点的章节号
        if re.search(r'\b\d{4}\b', line) and not re.match(r'^\d+(\.\d+)+', line):
            return False
        return True

    return False

def _extract_pdf(file_path: Path, original_filename: str = "") -> tuple[list[dict], dict]:
    pages: list[dict] = []
    tables: list[dict] = []
    doc = fitz.open(file_path)
    if len(doc) > MAX_PDF_PAGES:
        doc.close()
        raise HTTPException(413, f"PDF 页数超过限制（{MAX_PDF_PAGES} 页）")
    meta = dict(doc.metadata or {})
    for index, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        method = "text"
        quality = min(1.0, len(text.strip()) / 300.0)
        if len(text.strip()) < 45:
            engine = _get_ocr()
            if engine:
                temp_name = None
                try:
                    pix = page.get_pixmap(dpi=180, alpha=False)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        temp_name = tmp.name
                    pix.save(temp_name)
                    result, _ = engine(temp_name)
                    ocr_text = "\n".join(str(item[1]) for item in (result or []) if len(item) > 1)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text, method = ocr_text, "ocr"
                    quality = min(1.0, len(text.strip()) / 300.0) * (0.78 if method == "ocr" else 1.0)
                except Exception as exc:
                    log.warning("OCR failed on page %s: %s", index, exc)
                finally:
                    if temp_name and os.path.exists(temp_name):
                        os.remove(temp_name)
        pages.append({"page": index, "text": text, "method": method, "quality": round(quality, 3)})
        # PyMuPDF table extraction is optional and version-dependent.
        try:
            found = page.find_tables()
            for t_index, table in enumerate(found.tables or []):
                rows = table.extract()
                if rows:
                    tables.append({"page": index, "table_index": t_index, "rows": rows})
        except Exception:
            pass
    doc.close()

    first_text = "\n".join(p["text"] for p in pages[:2])
    whole_text = "\n".join(p["text"] for p in pages)
    title = (meta.get("title") or "").strip()
    if not title or title.lower() in {"untitled", "unknown"}:
        title = next((line.strip() for line in first_text.splitlines() if 12 <= len(line.strip()) <= 180), "未识别标题")
    doi_match = DOI_RE.search("\n".join(p["text"] for p in pages[:8]))
    arxiv_match = ARXIV_RE.search(original_filename + " " + first_text[:500])
    abstract_match = re.search(r"(?is)\babstract\b\s*[:\n]?(.{80,2500}?)(?:\n\s*(?:1\.?\s*)?introduction\b|\n\s*keywords\b)", first_text)
    author_lines = [x.strip() for x in first_text.splitlines()[:25] if "@" in x or re.search(r"\b(university|institute|laboratory|研究院|大学)\b", x, re.I)]
    headings = []
    sections = []
    known_headings = {"abstract", "introduction", "related work", "method", "methods", "experiments", "results", "discussion", "conclusion", "future work", "references", "appendix"}
    for page_item in pages:
        for line in page_item["text"].splitlines():
            value = line.strip()
            if 3 <= len(value) <= 90 and _is_valid_heading(value):
                if value not in headings:
                    headings.append(value)
                if not any(section["title"] == value for section in sections):
                    number_match = re.match(r"^(\d+(?:\.\d+)*)\s+", value)
                    level = number_match.group(1).count(".") + 1 if number_match else 1
                    sections.append({"title": value, "page": page_item["page"], "level": min(level, 4)})
    references_match = re.search(r"(?is)(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)(.*)$", whole_text)
    references_text = references_match.group(1) if references_match else ""
    reference_dois = list(dict.fromkeys(x.rstrip(".,;") for x in DOI_RE.findall(references_text)))[:300]
    reference_entries = [x.strip() for x in re.split(r"\n+", references_text) if len(x.strip()) > 25][:100]
    figure_captions = [x.strip() for x in whole_text.splitlines() if re.match(r"(?i)^\s*(figure|fig\.)\s+\d+[.:]", x.strip())][:100]
    metadata = {
        "title": title[:300],
        "author_affiliation_hints": author_lines[:8],
        "abstract": abstract_match.group(1).strip()[:2500] if abstract_match else "",
        "doi": doi_match.group(0).rstrip(".,;") if doi_match else "",
        "arxiv_id": arxiv_match.group(1) if arxiv_match else "",
        "arxiv_version": int(arxiv_match.group(2)) if arxiv_match and arxiv_match.group(2) else None,
        "headings": headings[:250],
        "sections": sections[:250],
        "reference_dois": reference_dois,
        "references_preview": reference_entries[:80],
        "figure_captions": figure_captions,
        "tables": tables[:80],
        "ocr_pages": [p["page"] for p in pages if p["method"] == "ocr"],
        "low_text_pages": [p["page"] for p in pages if len(p["text"].strip()) < 25],
    }
    return pages, metadata


def _clean_index_text(text: str) -> str:
    """
    Remove a known Google copyright-permission notice from retrieval text.
    Original PDF page text is preserved in the pages table.
    """
    if not text:
        return ""

    noise_patterns = [
        (
            r"(?is)Provided\s+proper\s+attribution\s+is\s+provided,"
            r"\s*Google\s+hereby\s+grants\s+permission\s+to"
            r"\s*reproduce\s+the\s+tables\s+and\s+figures\s+in\s+this\s+paper"
            r"\s*solely\s+for\s+use\s+in\s+journalistic\s+or"
            r"\s*scholarly\s+works\.?"
        ),
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text)

    return text.strip()


def _chunk_pages(pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    current_section = ""

    for p in pages:
        # Clean retrieval text only; do not modify the original page content.
        page_text = _clean_index_text(p.get("text", ""))

        paragraphs = [
            x.strip()
            for x in re.split(
                r"\n{1,}|(?<=[.!?])\s{2,}",
                page_text
            )
            if x.strip()
        ]

        buffer = ""

        for paragraph in paragraphs:
            if len(paragraph) < 120:
                heading_match = SECTION_RE.match(paragraph)

                if heading_match and (
                    paragraph.isupper()
                    or re.match(r"^\d+(?:\.\d+)*\s", paragraph)
                ):
                    current_section = paragraph[:120]

            if len(buffer) + len(paragraph) + 1 <= CHUNK_CHARS:
                buffer = f"{buffer}\n{paragraph}".strip()
            else:
                if len(buffer) >= 80:
                    chunks.append({
                        "id": uuid.uuid4().hex,
                        "page": p["page"],
                        "section": current_section,
                        "text": buffer,
                    })

                tail = buffer[-CHUNK_OVERLAP:] if buffer else ""
                buffer = (tail + "\n" + paragraph).strip()

        if len(buffer) >= 40:
            chunks.append({
                "id": uuid.uuid4().hex,
                "page": p["page"],
                "section": current_section,
                "text": buffer,
            })

    return chunks

# -----------------------------------------------------------------------------
# Optional dense embeddings + BM25 sparse retrieval
# -----------------------------------------------------------------------------
def _get_local_embedding_model():
    global _local_embedding_model
    if _local_embedding_model is None and LOCAL_EMBEDDING_MODEL and SentenceTransformer:
        _local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_embedding_model


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if embedding_client:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 64):
            response = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=texts[offset:offset + 64])
            vectors.extend([item.embedding for item in response.data])
        return vectors
    model = _get_local_embedding_model()
    if model:
        arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [row.astype(float).tolist() for row in arr]
    return []


def _get_reranker():
    global _reranker_model
    if _reranker_model is None and RERANKER_MODEL:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder(RERANKER_MODEL)
        except Exception as exc:
            log.warning("Unable to load reranker; RRF ranking remains active: %s", exc)
    return _reranker_model


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _bm25_scores(query: str, docs: list[str]) -> list[float]:
    q = _retrieval_tokens(query)
    token_docs = [_retrieval_tokens(x) for x in docs]
    n = len(token_docs)
    if not n or not q:
        return [0.0] * n
    avgdl = sum(len(x) for x in token_docs) / max(n, 1)
    df = Counter(t for doc in token_docs for t in set(doc))
    scores = []
    k1, b = 1.5, 0.75
    for doc in token_docs:
        freq = Counter(doc); dl = len(doc); score = 0.0
        for token in q:
            f = freq.get(token, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
        scores.append(score)
    return scores


def _load_user_chunks(user_id: str, paper_ids: Optional[list[str]] = None) -> list[dict]:
    with db() as conn:
        params: list[Any] = [user_id]
        where = "p.user_id=?"
        if paper_ids:
            marks = ",".join("?" for _ in paper_ids)
            where += f" AND p.id IN ({marks})"
            params.extend(paper_ids)
        rows = conn.execute(f"""
            SELECT c.id,c.paper_id,c.page_no,c.section,c.text,c.embedding_json,
                   p.filename,p.metadata_json
            FROM chunks c JOIN papers p ON p.id=c.paper_id
            WHERE {where}
        """, params).fetchall()
    return [dict(r) for r in rows]


def retrieve(user_id: str, query: str, paper_ids: Optional[list[str]] = None, top_k: int = DEFAULT_TOP_K, force_return: bool = False, rerank_query: Optional[str] = None) -> list[dict]:
    _retrieve_started = time.perf_counter()
    _load_started = time.perf_counter()
    rows = _load_user_chunks(user_id, paper_ids)
    log.info("[PERF][RETRIEVE] load_chunks=%.2fs rows=%d", time.perf_counter() - _load_started, len(rows))
    if not rows:
        return []
    if not _retrieval_tokens(query):
        return []
    _bm25_started = time.perf_counter()
    sparse = _bm25_scores(query, [r["text"] for r in rows])
    log.info("[PERF][RETRIEVE] bm25=%.2fs", time.perf_counter() - _bm25_started)

    dense_query: list[float] = []
    _embedding_started = time.perf_counter()
    try:
        dense_query = _embed_texts([query])[0] if (embedding_client or LOCAL_EMBEDDING_MODEL) else []
    except Exception as exc:
        log.warning("Embedding query failed; using BM25 only: %s", exc)
    log.info("[PERF][RETRIEVE] query_embedding=%.2fs", time.perf_counter() - _embedding_started)
    dense_scores = []
    for row in rows:
        vector = json.loads(row["embedding_json"]) if row.get("embedding_json") else []
        dense_scores.append(_cosine(dense_query, vector) if dense_query and vector else 0.0)
    # Reciprocal rank fusion avoids mixing incompatible score scales.
    sparse_rank = {idx: rank for rank, idx in enumerate(sorted(range(len(rows)), key=lambda i: sparse[i], reverse=True), 1)}
    dense_rank = {idx: rank for rank, idx in enumerate(sorted(range(len(rows)), key=lambda i: dense_scores[i], reverse=True), 1)}
    has_dense = bool(dense_query and any(dense_scores))
    fused = []
    for i, row in enumerate(rows):
        score = 1 / (60 + sparse_rank[i])
        if has_dense:
            score += 1 / (60 + dense_rank[i])
        fused.append((score, i))
    fused.sort(reverse=True)
    fused_score_by_index = {idx: score for score, idx in fused}
    # Never return arbitrary top-ranked chunks when the query has no lexical
    # or semantic overlap. This abstention is important for evidence-grounded QA.
    eligible = [
        idx for idx in range(len(rows))
        if sparse[idx] > 0 or (has_dense and dense_scores[idx] >= 0.18)
    ]
    if not eligible:
        return []
    # Expand candidates for cross-lingual CJK queries when
    # no meaningful CJK query terms occur in the user's corpus.
    query_tokens = _retrieval_tokens(query)
    has_cjk_query = any(
        "\u3400" <= char <= "\u9fff"
        for char in query
    )

    cjk_query_terms = [
        token for token in query_tokens
        if len(token) >= 2
        and any("\u3400" <= char <= "\u9fff" for char in token)
    ]

    corpus_tokens = {
        token
        for row in rows
        for token in _retrieval_tokens(row["text"])
    }

    has_cjk_lexical_match = bool(cjk_query_terms) and any(
        token in corpus_tokens
        for token in cjk_query_terms
    )

    expand_for_cross_lingual = (
        has_cjk_query and not has_cjk_lexical_match
    )

    candidate_cap = 60 if expand_for_cross_lingual else 40
    candidate_count = min(
        max(
            20,
            top_k * 4,
            60 if expand_for_cross_lingual else 20
        ),
        candidate_cap
    )

    log.info(
        "Retrieval candidates: %s, CJK query=%s, CJK lexical match=%s",
        candidate_count,
        has_cjk_query,
        has_cjk_lexical_match,
    )

    candidate_indices = sorted(
        eligible,
        key=lambda idx: fused_score_by_index[idx],
        reverse=True
    )[:candidate_count]
        # Save raw Cross-Encoder scores separately from RRF scores.
    rerank_score_by_index: dict[int, float] = {}

    _reranker_started = time.perf_counter()
    reranker = _get_reranker() if ENABLE_RERANKER else None

    if reranker and candidate_indices:
        try:
            import torch

            # ? Reranker ?????????????? query?
            effective_rerank_query = rerank_query or query
            pairs = [
                [effective_rerank_query, rows[idx]["text"][:1800]]
                for idx in candidate_indices
            ]

            # Keep raw logits for a more readable score range.
            # Disable progress output to keep backend logs clean.
            rerank_scores = reranker.predict(
                pairs,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
            )

            if len(rerank_scores) != len(candidate_indices):
                raise ValueError("重排序分数数量与候选片段数量不一致")

            scored_candidates = [
                (float(score), idx)
                for score, idx in zip(rerank_scores, candidate_indices)
            ]

            if not all(math.isfinite(score) for score, _ in scored_candidates):
                raise ValueError("重排序结果包含非有限数值")

            rerank_score_by_index = {
                idx: score
                for score, idx in scored_candidates
            }

            candidate_indices = [
                idx
                for score, idx in sorted(
                    scored_candidates,
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]

            # ?????????????????????????????
            if candidate_indices:
                top_rerank_score = rerank_score_by_index[candidate_indices[0]]
                # 动态阈值：针对论文的宏观总结/贡献类问题，跨语言打分天然偏低，需要放宽限制
                meta_keywords = [
                    chr(0x8d21) + chr(0x732e), chr(0x521b) + chr(0x65b0), chr(0x603b) + chr(0x7ed3),
                    chr(0x8bb2) + chr(0x4e86), chr(0x6458) + chr(0x8981), chr(0x6838) + chr(0x5fc3),
                    "contribution", "summary", "overview"
                ]
                is_meta = any(k in query.lower() for k in meta_keywords)
                effective_threshold = -10.0 if is_meta else RERANK_MIN_SCORE

                if not force_return and top_rerank_score < effective_threshold:
                    log.info("Abstaining: top rerank score %.4f < %.4f", top_rerank_score, effective_threshold)
                    return []

        except Exception as exc:
            log.warning(
                "Reranker failed; using hybrid ranking: %s",
                exc,
            )

    log.info(
        "[PERF][RETRIEVE] reranker=%.2fs candidates=%d enabled=%s",
        time.perf_counter() - _reranker_started,
        len(candidate_indices),
        bool(reranker),
    )

    selected_indices = candidate_indices[:max(1, min(top_k, 30))]

    result = []

    for idx in selected_indices:
        row = rows[idx]

        result.append({
            "evidence_id": f"E{len(result) + 1}",
            "chunk_id": row["id"],
            "paper_id": row["paper_id"],
            "filename": row["filename"],
            "page": row["page_no"],
            "section": row["section"],
            "quote": row["text"][:1200],

            # Original RRF fusion score; keep for compatibility.
            "score": round(
                float(fused_score_by_index.get(idx, 0.0)),
                6,
            ),

            # Raw Cross-Encoder score; None if reranking was unavailable.
            "rerank_score": (
                round(rerank_score_by_index[idx], 6)
                if idx in rerank_score_by_index
                else None
            ),
        })

    log.info(
        "[PERF][RETRIEVE] total=%.2fs returned=%d",
        time.perf_counter() - _retrieve_started,
        len(result),
    )
    return result



def _prepare_chunk_vectors(chunks: list[dict]) -> list[list[float]]:
    """在数据库事务开始前计算向量；向量服务不可用时保留 BM25 检索。"""
    vectors: list[list[float]] = []
    try:
        if chunks and (embedding_client or LOCAL_EMBEDDING_MODEL):
            vectors = _embed_texts([chunk["text"] for chunk in chunks])
    except Exception as exc:
        log.warning(
            "Dense indexing failed; sparse retrieval remains available: %s",
            exc,
        )
    return vectors


def _index_chunks(
    user_id: str,
    paper_id: str,
    chunks: list[dict],
    conn: Optional[sqlite3.Connection] = None,
    vectors: Optional[list[list[float]]] = None,
) -> None:
    """
    写入论文分块。

    如果传入 conn，则使用外部事务；
    如果没有传入 conn，则保持原来的独立数据库写入方式。
    """
    if vectors is None:
        vectors = _prepare_chunk_vectors(chunks)

    def insert_rows(target_conn: sqlite3.Connection) -> None:
        for i, chunk in enumerate(chunks):
            vector = vectors[i] if i < len(vectors) else None

            target_conn.execute(
                """
                INSERT INTO chunks(
                    id, paper_id, page_no, section, text,
                    embedding_json, created_at
                )
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    chunk["id"],
                    paper_id,
                    chunk["page"],
                    chunk["section"],
                    chunk["text"],
                    json.dumps(vector) if vector else None,
                    _iso(_now()),
                ),
            )

    if conn is not None:
        # 与论文记录、页面记录共用同一个事务。
        insert_rows(conn)
    else:
        # 保留原有调用方式，避免影响其他潜在调用者。
        with db() as target_conn:
            insert_rows(target_conn)


@app.post("/api/index/rebuild")
def rebuild_embeddings(user: dict = Depends(current_user)):
    """Rebuild vectors after an operator enables an embedding model in .env."""
    if embedding_client is None and not LOCAL_EMBEDDING_MODEL:
        raise HTTPException(503, "请先配置 EMBEDDING_* 或 LOCAL_EMBEDDING_MODEL")
    if embedding_client is None and _get_local_embedding_model() is None:
        raise HTTPException(503, "本地向量模型未加载，请安装 sentence-transformers 并检查模型名称")
    with db() as conn:
        rows = conn.execute(
            "SELECT c.id,c.text FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE p.user_id=? ORDER BY c.created_at",
            (user["id"],),
        ).fetchall()
    updated = 0
    for offset in range(0, len(rows), 64):
        batch = rows[offset:offset + 64]
        try:
            vectors = _embed_texts([r["text"] for r in batch])
        except Exception as exc:
            raise HTTPException(502, f"向量重建失败：{exc}")
        with db() as conn:
            for row, vector in zip(batch, vectors):
                conn.execute("UPDATE chunks SET embedding_json=? WHERE id=?", (json.dumps(vector), row["id"]))
                updated += 1
    return {"updated_chunks": updated, "message": "向量索引重建完成"}


# -----------------------------------------------------------------------------
# Common paper helpers
# -----------------------------------------------------------------------------
def _paper_rows(user_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute("SELECT id,filename,page_count,metadata_json,created_at FROM papers WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    result = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        result.append({"id": row["id"], "filename": row["filename"], "page_count": row["page_count"], "title": metadata.get("title", ""), "created_at": row["created_at"]})
    return result


def _get_paper(user_id: str, paper_id: str) -> dict:
    with db() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id=? AND user_id=?", (paper_id, user_id)).fetchone()
    if not row:
        raise HTTPException(404, "论文不存在")
    return dict(row)


def validate_evidence_sources(sources: list[dict]) -> dict:
    """Validate evidence structure and evidence ID uniqueness."""
    import re

    errors = []
    warnings = []
    seen = set()
    valid_ids = []

    if not isinstance(sources, list) or not sources:
        return {
            "valid": False,
            "errors": ["No evidence sources provided"],
            "warnings": [],
            "evidence_ids": [],
        }

    required = ("evidence_id", "filename", "page", "section", "quote")

    for index, item in enumerate(sources):
        if not isinstance(item, dict):
            errors.append(f"Source {index + 1} is not an object")
            continue

        missing = [key for key in required if key not in item]
        if missing:
            errors.append(
                f"Source {index + 1} missing fields: {', '.join(missing)}"
            )
            continue

        evidence_id = str(item.get("evidence_id", "")).strip()
        if not re.fullmatch(r"E[1-9]\d*", evidence_id):
            errors.append(f"Invalid evidence ID: {evidence_id!r}")
        elif evidence_id in seen:
            errors.append(f"Duplicate evidence ID: {evidence_id}")
        else:
            seen.add(evidence_id)
            valid_ids.append(evidence_id)

        if not str(item.get("filename") or "").strip():
            errors.append(f"{evidence_id or index + 1}: empty filename")

        try:
            page = int(item.get("page"))
            if page < 1:
                errors.append(f"{evidence_id or index + 1}: invalid page")
        except (TypeError, ValueError):
            errors.append(f"{evidence_id or index + 1}: invalid page")

        if not str(item.get("quote") or "").strip():
            errors.append(f"{evidence_id or index + 1}: empty quote")

        if item.get("section") is None:
            errors.append(f"{evidence_id or index + 1}: section is null")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "evidence_ids": valid_ids,
    }


def validate_answer_citations(answer: str, evidence_ids: list[str]) -> dict:
    """Check whether answer citation IDs exist in the supplied evidence."""
    import re

    allowed = set(evidence_ids)
    citations = re.findall(r"\[(E[1-9]\d*)\]", answer or "")
    unknown = sorted(set(citations) - allowed)
    warnings = []

    if not citations:
        warnings.append("No evidence citations found in the answer")

    if unknown:
        warnings.append("Answer contains unknown evidence IDs")

    return {
        "valid": not unknown,
        "citation_count": len(citations),
        "used_evidence_ids": sorted(set(citations)),
        "unknown_evidence_ids": unknown,
        "warnings": warnings,
    }



def validate_answer_semantic_support(answer: str, sources: list[dict]) -> dict:
    """Use an independent LLM pass to verify each factual claim against cited evidence."""
    allowed_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in sources
        if isinstance(item, dict) and item.get("evidence_id")
    }

    if not answer.strip() or not allowed_ids:
        return {
            "valid": False,
            "semantic_support_checked": False,
            "claims": [],
            "errors": ["Answer or evidence is empty"],
            "warnings": ["Semantic support could not be checked"],
        }

    prompt = f"""You are a strict academic evidence auditor.

Review every factual claim in the answer against the supplied paper evidence.
Do not assume that a citation supports a claim merely because its ID exists.

For each factual claim, choose exactly one verdict:
- supported: directly supported by the cited evidence
- partially_supported: only part of the claim is supported
- unsupported: unrelated, contradicted, or not supported by the evidence
- insufficient_evidence: the supplied evidence is insufficient to decide

Requirements:
1. Split the answer into independently verifiable factual claims.
2. Return the evidence IDs that actually support each claim.
3. Never invent evidence IDs.
4. Do not treat speculation as an explicit paper conclusion.
5. Verify formulas, numbers, comparisons, and causal claims.
6. A claim without direct evidence must not be marked supported.
7. Return valid JSON only. Do not use Markdown fences.

Required JSON structure:
{{
  "claims": [
    {{
      "claim": "A factual claim from the answer",
      "verdict": "supported",
      "evidence_ids": ["E1"],
      "reason": "Brief explanation based on the source"
    }}
  ]
}}

PAPER EVIDENCE:
{_source_context(sources)}

ANSWER TO REVIEW:
{answer}
"""

    result = llm_json(
        prompt,
        system="You are a reliable JSON output generator. Return valid JSON only.",
    )

    if not isinstance(result, dict) or not isinstance(result.get("claims"), list):
        return {
            "valid": False,
            "semantic_support_checked": False,
            "claims": [],
            "errors": ["Semantic validator returned an invalid structure"],
            "warnings": ["Semantic support could not be checked"],
        }

    allowed_verdicts = {
        "supported",
        "partially_supported",
        "unsupported",
        "insufficient_evidence",
    }

    claims = []
    errors = []
    warnings = []

    for index, item in enumerate(result["claims"]):
        if not isinstance(item, dict):
            errors.append(f"Claim {index + 1} is not an object")
            continue

        verdict = str(item.get("verdict", "")).strip()
        claim_text = str(item.get("claim", "")).strip()
        evidence_ids = item.get("evidence_ids", [])

        if verdict not in allowed_verdicts:
            errors.append(f"Claim {index + 1} has an invalid verdict")
            continue

        if not claim_text:
            errors.append(f"Claim {index + 1} is empty")
            continue

        if not isinstance(evidence_ids, list):
            evidence_ids = []
            errors.append(f"Claim {index + 1} has invalid evidence_ids")

        evidence_ids = [str(x).strip() for x in evidence_ids]
        unknown_ids = sorted(set(evidence_ids) - allowed_ids)

        if unknown_ids:
            errors.append(
                f"Claim {index + 1} references unknown evidence IDs: {unknown_ids}"
            )
            verdict = "unsupported"

        claims.append({
            "claim": claim_text,
            "verdict": verdict,
            "evidence_ids": evidence_ids,
            "reason": str(item.get("reason", "")).strip(),
        })

        if verdict != "supported":
            warnings.append(
                f"Claim {index + 1}: {verdict}"
            )

    if not claims:
        errors.append("No verifiable factual claims were returned")

    return {
        "valid": bool(claims) and not errors and all(
            item["verdict"] == "supported" for item in claims
        ),
        "semantic_support_checked": True,
        "claim_count": len(claims),
        "supported_count": sum(
            item["verdict"] == "supported" for item in claims
        ),
        "claims": claims,
        "errors": errors,
        "warnings": warnings,
    }


def _source_context(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[{s['evidence_id']}] 文献={s['filename']}；页码={s['page']}；章节={s['section']}\n{s['quote']}"
        for s in sources
    )


def _stream_response(generator):
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


# -----------------------------------------------------------------------------
# Health, paper CRUD, batch upload, deduplication and version comparison
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_ready": bool(llm_client),
        "ocr_ready": RapidOCR is not None,
        "dense_retrieval_ready": bool(embedding_client or _local_embedding_model is not None),
        "dense_retrieval_configured": bool(embedding_client or LOCAL_EMBEDDING_MODEL),
        "database": "sqlite-wal",
        "version": app.version,
    }




def _upload_bytes(user: dict, filename: str, content: bytes) -> dict:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "仅支持 PDF 文件")

    if not content or len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"文件为空或超过 {MAX_UPLOAD_MB} MB")

    if not content.startswith(b"%PDF"):
        raise HTTPException(415, "文件内容不是有效 PDF")

    sha = hashlib.sha256(content).hexdigest()

    # 先进行 SHA256 精确重复检测。
    with db() as conn:
        exact = conn.execute(
            """
            SELECT id, filename
            FROM papers
            WHERE user_id=? AND sha256=?
            """,
            (user["id"], sha),
        ).fetchone()

    if exact:
        return {
            "paper_id": exact["id"],
            "filename": exact["filename"],
            "duplicates": [
                {
                    "paper_id": exact["id"],
                    "filename": exact["filename"],
                    "similarity": 1.0,
                    "suggestion": "文件内容完全相同",
                }
            ],
            "already_exists": True,
        }

    paper_id = uuid.uuid4().hex[:12]
    safe_name = f"{paper_id}.pdf"
    path = PAPER_DIR / safe_name
    upload_started = time.perf_counter()

    try:
        # 1. 保存 PDF 文件。
        path.write_bytes(content)

        # 2. 提取 PDF 文本和元数据，保留原有 OCR 逻辑。
        _pdf_started = time.perf_counter()
        pages, metadata = _extract_pdf(
            path,
            original_filename=filename,
        )
        log.info(
            "[PERF][UPLOAD] pdf_extract=%.2fs pages=%d ocr_pages=%d",
            time.perf_counter() - _pdf_started,
            len(pages),
            len(metadata.get("ocr_pages", [])),
        )

        full_text = "\n\n".join(
            page["text"] for page in pages
        )

        # 3. 检查与现有论文的相似度和 arXiv 版本。
        _dedup_started = time.perf_counter()
        with db() as conn:
            candidates = conn.execute(
                """
                SELECT id, filename, full_text, metadata_json
                FROM papers
                WHERE user_id=?
                """,
                (user["id"],),
            ).fetchall()

        duplicates = []

        for old in candidates:
            old_meta = json.loads(old["metadata_json"] or "{}")

            same_arxiv = bool(
                metadata.get("arxiv_id")
                and metadata.get("arxiv_id") == old_meta.get("arxiv_id")
            )

            ratio = _shingle_similarity(
                full_text,
                old["full_text"],
            )

            # 即使 arXiv ID 相同，文本相似度也至少要有 10% (0.1) 才能算同一篇论文。
            # 防止正文里“提及”了某个 ID 就被误判。
            is_same_paper = (same_arxiv and ratio >= 0.10) or (ratio >= 0.18)

            if is_same_paper:
                old_version = old_meta.get("arxiv_version") or 0
                new_version = metadata.get("arxiv_version") or 0

                if same_arxiv and ratio >= 0.10:
                    suggestion = "检测到同一 arXiv 论文的不同版本"
                else:
                    suggestion = "正文语义片段高度相似，建议核对版本"

                if same_arxiv and old_version > new_version:
                    suggestion += "；知识库现有版本更新，建议保留现有版本"
                elif same_arxiv and new_version > old_version:
                    suggestion += "；新上传版本更新，可考虑替换旧版本"

                # 只有真正判定为同一篇论文，才计算高相似度
                calc_sim = ratio if ratio > 0 else 0.0
                if same_arxiv and ratio >= 0.10:
                    calc_sim = max(calc_sim, 0.99)

                duplicates.append(
                    {
                        "paper_id": old["id"],
                        "filename": old["filename"],
                        "similarity": round(calc_sim, 3),
                        "suggestion": suggestion,
                        "existing_version": old_version,
                        "uploaded_version": new_version,
                    }
                )

        # 4. 先生成 chunks 和向量。
        #    Embedding 计算放在数据库事务之外，避免长时间占用写锁。
        log.info(
            "[PERF][UPLOAD] dedup=%.2fs existing_papers=%d duplicates=%d",
            time.perf_counter() - _dedup_started,
            len(candidates),
            len(duplicates),
        )

        _chunk_started = time.perf_counter()
        chunks = _chunk_pages(pages)
        log.info(
            "[PERF][UPLOAD] chunking=%.2fs chunks=%d",
            time.perf_counter() - _chunk_started,
            len(chunks),
        )

        _embedding_started = time.perf_counter()
        vectors = _prepare_chunk_vectors(chunks)
        log.info(
            "[PERF][UPLOAD] embedding=%.2fs chunks=%d vectors=%d",
            time.perf_counter() - _embedding_started,
            len(chunks),
            len(vectors),
        )

        # 5. 提前组装响应内容，避免数据库提交后再执行复杂处理。
        result = {
            "paper_id": paper_id,
            "filename": filename,
            "page_count": len(pages),
            "duplicates": duplicates,
            "metadata": metadata,
            "chunk_count": len(chunks),
            "warnings": (
                [f"OCR 处理了 {len(metadata['ocr_pages'])} 页"]
                if metadata["ocr_pages"]
                else []
            ) + (
                [f"{len(metadata['low_text_pages'])} 页文本较少，请抽查"]
                if metadata["low_text_pages"]
                else []
            ),
        }

        # 6. 关键修复：
        #    papers、pages、chunks 全部使用同一个数据库事务。
        #    任意一步失败，db() 会统一 rollback。
        _db_write_started = time.perf_counter()
        with db() as conn:
            conn.execute(
                """
                INSERT INTO papers(
                    id, user_id, filename, file_path, sha256,
                    full_text, page_count, metadata_json, created_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    paper_id,
                    user["id"],
                    Path(filename).name,
                    str(path),
                    sha,
                    full_text,
                    len(pages),
                    json.dumps(metadata, ensure_ascii=False),
                    _iso(_now()),
                ),
            )

            for page in pages:
                conn.execute(
                    """
                    INSERT INTO pages(
                        paper_id, page_no, text,
                        extraction_method, quality
                    )
                    VALUES(?,?,?,?,?)
                    """,
                    (
                        paper_id,
                        page["page"],
                        page["text"],
                        page["method"],
                        page["quality"],
                    ),
                )

            # 传入当前 conn，不再单独开启第二个数据库事务。
            _index_chunks(
                user["id"],
                paper_id,
                chunks,
                conn=conn,
                vectors=vectors,
            )

        # 只有所有数据库记录成功提交后，才返回上传成功。
        log.info(
            "[PERF][UPLOAD] database_write=%.2fs",
            time.perf_counter() - _db_write_started,
        )
        log.info(
            "[PERF][UPLOAD] total=%.2fs pages=%d chunks=%d",
            time.perf_counter() - upload_started,
            len(pages),
            len(chunks),
        )
        return result

    except Exception:
        # 异常补偿：如果数据库事务已经提交但后续发生异常，
        # 尝试删除该论文记录，pages 和 chunks 会通过外键级联删除。
        try:
            with db() as cleanup_conn:
                cleanup_conn.execute(
                    """
                    DELETE FROM papers
                    WHERE id=? AND user_id=?
                    """,
                    (paper_id, user["id"]),
                )
        except Exception:
            log.exception(
                "Upload cleanup failed for paper_id=%s",
                paper_id,
            )

        # 清理刚刚写入的 PDF 文件。
        try:
            if path.exists():
                path.unlink()
        except OSError:
            log.exception(
                "Failed to remove uploaded PDF: %s",
                path,
            )

        # 保留原始异常，便于定位真正的错误。
        raise



@app.get("/api/papers")
def list_papers(user: dict = Depends(current_user)):
    return _paper_rows(user["id"])


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...), user: dict = Depends(current_user)):
    content = await file.read()
    # PDF parsing/OCR is CPU-bound; offload it so the ASGI event loop can keep
    # serving other requests while a 500+ page document is being indexed.
    return await run_in_threadpool(_upload_bytes, user, file.filename or "upload.pdf", content)


@app.post("/api/papers/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...), user: dict = Depends(current_user)):
    if len(files) > 30:
        raise HTTPException(413, "单次最多上传 30 篇论文")
    results = []
    for file in files:
        try:
            content = await file.read()
            result = await run_in_threadpool(_upload_bytes, user, file.filename or "upload.pdf", content)
            results.append({"ok": True, **result})
        except HTTPException as exc:
            results.append({"ok": False, "filename": file.filename, "error": exc.detail})
    return {"items": results}


@app.get("/api/papers/{paper_id}")
def get_paper(
    paper_id: str,
    start_page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=60),
    user: dict = Depends(current_user),
):
    paper = _get_paper(user["id"], paper_id)
    with db() as conn:
        page_rows = conn.execute(
            "SELECT page_no,text,extraction_method,quality FROM pages WHERE paper_id=? AND page_no>=? ORDER BY page_no LIMIT ?",
            (paper_id, start_page, page_size),
        ).fetchall()
    metadata = json.loads(paper["metadata_json"] or "{}")
    # Avoid loading a 541-page full text in one browser request.
    return {
        "pid": paper_id,
        "id": paper_id,
        "filename": paper["filename"],
        "page_count": paper["page_count"],
        "start_page": start_page,
        "page_size": page_size,
        "has_more": start_page + len(page_rows) <= paper["page_count"] and bool(page_rows),
        "pages": [{"page": r["page_no"], "text": r["text"], "method": r["extraction_method"], "quality": r["quality"]} for r in page_rows],
        "metadata": metadata,
        "full_text": paper["full_text"] if paper["page_count"] <= 40 else "",
    }


@app.get("/api/papers/{paper_id}/pages")
def get_paper_pages(paper_id: str, start_page: int = 1, page_size: int = 20, user: dict = Depends(current_user)):
    return get_paper(paper_id, start_page=max(1, start_page), page_size=max(1, min(60, page_size)), user=user)


@app.get("/api/papers/{paper_id}/outline")
def get_paper_outline(paper_id: str, user: dict = Depends(current_user)):
    paper = _get_paper(user["id"], paper_id)
    metadata = json.loads(paper["metadata_json"] or "{}")
    return {"paper_id": paper_id, "title": metadata.get("title", paper["filename"]), "headings": metadata.get("headings", []), "sections": metadata.get("sections", []), "references": metadata.get("references_preview", []), "reference_dois": metadata.get("reference_dois", []), "figure_captions": metadata.get("figure_captions", [])}


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: str, user: dict = Depends(current_user)):
    paper = _get_paper(user["id"], paper_id)
    with db() as conn:
        conn.execute("DELETE FROM papers WHERE id=? AND user_id=?", (paper_id, user["id"]))
    path = Path(paper["file_path"])
    if path.exists(): path.unlink()
    return {"message": "删除成功"}


@app.post("/api/papers/{paper_id}/resolve-duplicate")
def resolve_duplicate(paper_id: str, payload: dict, user: dict = Depends(current_user)):
    candidate_id = str(payload.get("candidate_id", ""))
    action = str(payload.get("action", "keep_both"))
    if action not in {"keep_new", "keep_existing", "keep_both"}:
        raise HTTPException(400, "action 仅支持 keep_new / keep_existing / keep_both")
    new_paper = _get_paper(user["id"], paper_id)
    old_paper = _get_paper(user["id"], candidate_id)
    if action == "keep_new":
        with db() as conn: conn.execute("DELETE FROM papers WHERE id=?", (candidate_id,))
        old_path = Path(old_paper["file_path"])
        if old_path.exists(): old_path.unlink()
        return {"message": "已保留新版本", "kept": paper_id}
    if action == "keep_existing":
        with db() as conn: conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        new_path = Path(new_paper["file_path"])
        if new_path.exists(): new_path.unlink()
        return {"message": "已保留知识库原版本", "kept": candidate_id}
    return {"message": "已保留两个版本", "kept": [paper_id, candidate_id]}


@app.get("/api/papers/{paper_id}/diff/{candidate_id}")
def compare_paper_versions(paper_id: str, candidate_id: str, user: dict = Depends(current_user)):
    new_paper = _get_paper(user["id"], paper_id)
    old_paper = _get_paper(user["id"], candidate_id)
    old_lines = old_paper["full_text"].splitlines()
    new_lines = new_paper["full_text"].splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines[:12000], b=new_lines[:12000], autojunk=False)
    added, removed, changed = 0, 0, 0
    excerpts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
            excerpts.extend("+ " + x[:240] for x in new_lines[j1:j2][:5])
        elif tag == "delete":
            removed += i2 - i1
            excerpts.extend("- " + x[:240] for x in old_lines[i1:i2][:5])
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)
            excerpts.extend("- " + x[:240] for x in old_lines[i1:i2][:3])
            excerpts.extend("+ " + x[:240] for x in new_lines[j1:j2][:3])
        if len(excerpts) >= 24: break
    summary = f"相对库中版本，文本差异约为：新增 {added} 行、删除 {removed} 行、替换约 {changed} 行。该统计基于文本抽取结果，不等同于版面或语义差异。"
    if llm_client and excerpts:
        try:
            prompt = (
                "请概括两版论文的实质变化，区分新增方法、实验、结果、修订措辞。若证据不足，不要推测。\n"
                f"旧版本：{old_paper['filename']}\n新版本：{new_paper['filename']}\n差异片段：\n"
                + "\n".join(excerpts[:24])
            )
            summary = llm_text(
                prompt,
                system="你是学术论文版本差异审阅专家。",
                temperature=0,
            )
        except Exception as exc:
            log.warning("Version diff LLM summary failed: %s", exc)
    return {"old_filename": old_paper["filename"], "new_filename": new_paper["filename"], "added_lines": added, "removed_lines": removed, "changed_lines": changed, "summary": summary, "diff_excerpt": excerpts[:24]}


@app.get("/api/papers/{paper_id}/download")
def download_paper(paper_id: str, user: dict = Depends(current_user)):
    paper = _get_paper(user["id"], paper_id)
    return FileResponse(paper["file_path"], filename=paper["filename"], media_type="application/pdf")


@app.get("/api/media/{filename}")
def get_media(filename: str, user: dict = Depends(current_user)):
    if not re.fullmatch(r"chart_[0-9a-f]{14}\.png", filename):
        raise HTTPException(404, "?????")

    with db() as conn:
        row = conn.execute(
            "SELECT user_id FROM media_assets WHERE filename=?",
            (filename,),
        ).fetchone()

    if not row or row["user_id"] != user["id"]:
        raise HTTPException(404, "?????")

    path = MEDIA_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "?????")

    return FileResponse(path, media_type="image/png")


# -----------------------------------------------------------------------------
# QA with evidence IDs and multi-paper retrieval
# -----------------------------------------------------------------------------
@app.post("/api/qa/search")
def qa_search(payload: dict, user: dict = Depends(current_user)):
    question = str(payload.get("question", "")).strip()
    paper_ids = payload.get("paper_ids")
    if not question: raise HTTPException(400, "问题不能为空")
    return {"sources": retrieve(user["id"], question, paper_ids=paper_ids, top_k=int(payload.get("top_k", DEFAULT_TOP_K)))}


@app.post("/api/qa/stream")
def qa_stream(payload: dict, user: dict = Depends(current_user)):
    paper_id = payload.get("paper_id")
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    if paper_id:
        _get_paper(user["id"], str(paper_id))

    rerank_query = _build_bilingual_rerank_query(question)

    sources = retrieve(
        user["id"],
        question,
        paper_ids=[paper_id] if paper_id else None,
        top_k=DEFAULT_TOP_K,
        rerank_query=rerank_query,
    )

    def generate():
        yield _sse({"agent": "论文问答 Agent", "status": "running"})
        yield _sse({
            "agent": "论文问答 Agent",
            "status": "done",
            "sources": sources,
        })

        yield _sse({"agent": "论文问答 Agent", "status": "running"})
        # No retrieved evidence is a normal abstention, not a validation error.
        if not sources:
            answer = (
                "\u5f53\u524d\u8bba\u6587\u68c0\u7d22\u5230\u7684\u5185\u5bb9\u4e2d\u6ca1\u6709\u8db3\u591f\u8bc1\u636e\u786e\u5b9a\u4f5c\u8005\u5206\u522b\u5c45\u4f4f\u5728\u54ea\u4e2a\u57ce\u5e02\u3002"
                "\u8bba\u6587\u4e2d\u7684\u4f5c\u8005\u5355\u4f4d\u6216\u8054\u7cfb\u5730\u5740\u4e5f\u4e0d\u80fd\u76f4\u63a5\u8bc1\u660e\u5176\u5c45\u4f4f\u5730\u3002"
                "\u4e3a\u907f\u514d\u7f16\u9020\u4fe1\u606f\uff0c\u6682\u4e0d\u5bf9\u6b64\u95ee\u9898\u4f5c\u51fa\u63a8\u65ad\u3002"
            )
            yield _sse({
                "agent": "论文问答 Agent",
                "status": "warning",
                "abstained": True,
                "reason": "insufficient_evidence",
            })
            yield _sse({
                "agent": "论文问答 Agent",
                "text": answer,
            })
            yield _sse({
                "agent": "论文问答 Agent",
                "status": "done",
            })
            return

        source_check = validate_evidence_sources(sources)

        if not source_check["valid"]:
            yield _sse({
                "agent": "论文问答 Agent",
                "status": "error",
                "validation": {
                    "stage": "before_generation",
                    **source_check,
                },
            })
            answer = (
                "\u68c0\u7d22\u5230\u7684\u8bc1\u636e\u6570\u636e\u4e0d\u5b8c\u6574"
                "\uff0c\u6682\u65f6\u65e0\u6cd5\u751f\u6210\u53ef\u9760\u56de\u7b54"
                "\u3002\u8bf7\u68c0\u67e5\u8bba\u6587\u89e3\u6790\u7ed3\u679c\u3002"
            )
            yield _sse({"agent": "论文问答 Agent", "text": answer})
            yield _sse({"agent": "论文问答 Agent", "status": "done"})
            return

        yield _sse({
            "agent": "论文问答 Agent",
            "status": "done",
            "validation": {
                "stage": "before_generation",
                **source_check,
            },
        })

        yield _sse({"agent": "论文问答 Agent", "status": "running"})

        prompt = f"""You are an academic research assistant.
Answer in professional, clear Simplified Chinese.
Use ONLY the supplied evidence.

Rules:
1. Every factual claim must have directly supporting evidence labels such as [E1].
2. Preserve evidence labels exactly. Never invent evidence IDs.
3. Never invent authors, methods, datasets, metrics, page numbers, or conclusions.
4. If evidence is insufficient, clearly state that the retrieved evidence is insufficient.
5. Distinguish statements explicitly made by the paper from evidence-based analysis.
6. Do not cite evidence that does not support the claim.
7. Explain conflicting evidence instead of silently choosing one version.
8. Do not introduce technical details from outside the supplied evidence.
9. Avoid repeating the same information.
10. Adapt the answer structure to the question. Do not force irrelevant sections.
11. Only describe limitations when the supplied evidence supports them.

When relevant and supported, cover:
- Main contribution and research problem.
- Technical method, architecture, and mechanisms.
- Experimental datasets, metrics, results, and conditions.
- Research value and evidence-based analysis.
- Limitations supported by the evidence.
- Traceable evidence labels next to factual claims.

Evidence:
{_source_context(sources)}

Question:
{question}
"""

        answer_parts = []
        try:
            if not llm_client:
                answer = (
                    "\u5df2\u627e\u5230\u76f8\u5173\u539f\u6587\u8bc1\u636e"
                    "\u3002\u8bf7\u914d\u7f6e LLM_API_KEY \u540e"
                    "\u542f\u7528\u57fa\u4e8e\u8bc1\u636e\u7684\u81ea\u7136\u8bed\u8a00\u7efc\u5408\u56de\u7b54\u3002\n\n"
                )
                answer += "\n".join(
                    f"[{s['evidence_id']}] {s['filename']} "
                    f"\u7b2c{s['page']}\u9875\uff1a{s['quote'][:240]}"
                    for s in sources
                )
                answer_parts.append(answer)
                answer_text = answer

                answer_check = validate_answer_citations(
                    answer_text,
                    source_check["evidence_ids"],
                )
                semantic_check = {
                    "valid": False,
                    "semantic_support_checked": False,
                    "warnings": ["LLM is not configured; semantic validation skipped"],
                }

            else:
                stream = llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    stream=True,
                )

                # Buffer the complete answer. Do not expose unverified text.
                for chunk in stream:
                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices else None
                    )
                    if delta:
                        answer_parts.append(delta)

                answer_text = "".join(answer_parts)

                answer_check = validate_answer_citations(
                    answer_text,
                    source_check["evidence_ids"],
                )

                # Run semantic verification independently.
                try:
                    semantic_check = validate_answer_semantic_support(
                        answer_text,
                        sources,
                    )
                except Exception as validation_exc:
                    log.exception("Semantic evidence validation failed")
                    semantic_check = {
                        "valid": False,
                        "semantic_support_checked": False,
                        "errors": [str(validation_exc)],
                        "warnings": ["Semantic validation could not be completed"],
                    }

            citations_ok = (
                answer_check["valid"]
                and answer_check["citation_count"] > 0
            )
            semantic_ok = (
                semantic_check.get("semantic_support_checked") is True
                and semantic_check.get("valid") is True
            )

            final_valid = citations_ok and semantic_ok

            # Send the answer only after validation has finished.
            yield _sse({
                "agent": "\u8bba\u6587\u95ee\u7b54 Agent",
                "text": answer_text,
            })

            yield _sse({
                "agent": "\u8bba\u6587\u95ee\u7b54 Agent",
                "status": "done" if final_valid else "warning",
                "validation": {
                    "stage": "after_generation",
                    "valid": final_valid,
                    "citation_check": answer_check,
                    "semantic_check": semantic_check,
                    "semantic_support_checked": semantic_ok,
                },
            })

        except Exception as exc:
            log.exception("QA stream failed")
            yield _sse({
                "agent": "\u8bba\u6587\u95ee\u7b54 Agent",
                "status": "error",
                "text": f"\u95ee\u7b54\u751f\u6210\u5931\u8d25\uff1a{exc}",
            })

    return _stream_response(generate())


@app.post("/api/qa/compare")
def qa_compare(payload: dict, user: dict = Depends(current_user)):
    paper_ids = [str(x) for x in payload.get("paper_ids", [])][:12]
    question = str(payload.get("question", "请比较核心创新、方法和实验差异")).strip()
    if len(paper_ids) < 2: raise HTTPException(400, "至少选择两篇论文")
    for pid in paper_ids: _get_paper(user["id"], pid)
    # Retrieve evidence independently per paper; avoids truncating only the first N chars.
    per_paper = []
    for pid in paper_ids:
        hits = retrieve(user["id"], question, [pid], top_k=4, force_return=True)
        for hit in hits:
            hit["evidence_id"] = f"E{len(per_paper) + 1}"
            per_paper.append(hit)
    def generate():
        yield _sse({"agent": "逐篇检索 Agent", "status": "running"})
        yield _sse({"agent": "逐篇检索 Agent", "status": "done", "sources": per_paper})
        yield _sse({"agent": "跨文献分析 Agent", "status": "running"})
        if not per_paper:
            yield _sse({"agent": "跨文献分析 Agent", "text": "没有检索到足够的证据，请检查已选文献是否完成解析。"})
            yield _sse({"agent": "跨文献分析 Agent", "status": "done"})
            return
        prompt = f"""你是科研综述分析专家。请基于下面每篇论文独立检索到的证据回答：{question}
请按“论文、研究问题、核心方法、关键实验/结果、创新点、局限性”组织对比表，再总结演进关系。
每项事实必须标注 [E编号]；未找到的字段标注“原文证据不足”，不得推测。

{_source_context(per_paper)}
"""
        try:
            if llm_client:
                stream = llm_client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1, stream=True)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta: yield _sse({"agent": "跨文献分析 Agent", "text": delta})
            else:
                yield _sse({"agent": "跨文献分析 Agent", "text": "已完成逐篇证据检索。配置模型 API 后可生成结构化跨文献综合报告。"})
            yield _sse({"agent": "跨文献分析 Agent", "status": "done"})
        except Exception as exc:
            yield _sse({"agent": "跨文献分析 Agent", "status": "error", "text": str(exc)})
    return _stream_response(generate())


# -----------------------------------------------------------------------------
# Citation graph, centrality, Future Work and ArXiv discovery
# -----------------------------------------------------------------------------
def _normalized_title(value: str) -> str:
    value = _normal_text(value)
    value = re.sub(r"\b(arxiv|preprint|v\d+)\b", " ", value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)


def _build_graph_data(user_id: str) -> dict:
    papers = []
    with db() as conn:
        rows = conn.execute("SELECT id,filename,full_text,metadata_json,page_count FROM papers WHERE user_id=?", (user_id,)).fetchall()
    papers = [dict(r) for r in rows]
    graph = nx.DiGraph()
    for p in papers:
        metadata = json.loads(p["metadata_json"] or "{}")
        graph.add_node(p["id"], label=metadata.get("title") or Path(p["filename"]).stem, filename=p["filename"], doi=metadata.get("doi", ""))
    title_index = {}
    doi_index = {}
    for p in papers:
        metadata = json.loads(p["metadata_json"] or "{}")
        title_index[p["id"]] = _normalized_title(metadata.get("title") or Path(p["filename"]).stem)
        if metadata.get("doi"): doi_index[p["id"]] = metadata["doi"].lower()
    for source in papers:
        text = source["full_text"].lower()
        for target in papers:
            if source["id"] == target["id"]: continue
            target_doi = doi_index.get(target["id"])
            target_title = title_index.get(target["id"], "")
            matched = bool(target_doi and target_doi in text)
            if not matched and len(target_title) >= 18:
                matched = target_title in _normalized_title(text)
            if matched:
                graph.add_edge(source["id"], target["id"], relation="cites", evidence="DOI/title mention")
    if graph.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "metrics": {}}
    pagerank = nx.pagerank(graph) if graph.number_of_edges() else {n: 1 / graph.number_of_nodes() for n in graph.nodes}
    try: between = nx.betweenness_centrality(graph)
    except Exception: between = {n: 0 for n in graph.nodes}
    nodes = []
    for node, data in graph.nodes(data=True):
        nodes.append({"id": node, "name": data["label"], "filename": data["filename"], "size": 45 + pagerank[node] * 1000, "pagerank": round(pagerank[node], 6), "betweenness": round(between[node], 6), "in_degree": graph.in_degree(node), "out_degree": graph.out_degree(node)})
    edges = [{"source": a, "target": b, **data} for a, b, data in graph.edges(data=True)]
    ranked = sorted(nodes, key=lambda x: x["pagerank"], reverse=True)
    return {"nodes": nodes, "edges": edges, "metrics": {"paper_count": graph.number_of_nodes(), "citation_edges": graph.number_of_edges(), "top_foundational_papers": ranked[:5], "isolated_papers": [n["name"] for n in nodes if graph.degree(n["id"]) == 0]}}


@app.post("/api/graph/build")
def build_graph(user: dict = Depends(current_user)):
    return _build_graph_data(user["id"])


def _future_work(user_id: str) -> dict:
    with db() as conn:
        rows = conn.execute("SELECT id,filename,full_text,metadata_json,page_count FROM papers WHERE user_id=?", (user_id,)).fetchall()

    papers = [dict(r) for r in rows]
    if not papers:
        return {"items": [], "summary": "文献库为空。"}

    items = []
    all_quotes = []

    # 极其严格的学术提取正则：只找带有明确未来工作/局限性标志的完整句子，且必须是纯英文
    strong_pattern = re.compile(r'(?i)(?:[A-Z][^.]*(?:future work|limitation|we leave|remains an? (?:open|challenging)|further investigate|further exploration)[^.]*\.)')

    for p in papers:
        text_content = p["full_text"]
        # 如果全篇中文字符占比过高，直接跳过（只处理英文论文）
        cn_chars = len(re.findall(r'[\u4e00-\u9fa5]', text_content))
        if cn_chars > len(text_content) * 0.2:
            continue

        matches = strong_pattern.findall(text_content)

        unique_matches = []
        for m in matches:
            m = m.strip().replace("\n", " ")
            # 过滤掉含有 dataset/benchmark/table/figure 等不相干的句子
            if any(bad in m.lower() for bad in ["dataset", "benchmark", "table ", "figure ", "appendix"]):
                continue
            if len(m) > 30 and m not in unique_matches:
                unique_matches.append(m)

        unique_matches = unique_matches[:3] # 每篇最多保留3条高价值句子

        if unique_matches:
            meta = json.loads(p["metadata_json"] or "{}")
            title = meta.get("title") or p["filename"]

            # 尝试简单定位页码
            page_no = 1
            first_match_text = unique_matches[0]
            idx = text_content.find(first_match_text[:30])
            if idx != -1:
                # 粗略估算页码 (假设每页 3000 字符)
                page_no = max(1, idx // 3000 + 1)

            evidence = [{"page": page_no, "quote": q} for q in unique_matches]

            items.append({
                "paper_id": p["id"],
                "filename": p["filename"],
                "title": title,
                "future_work": unique_matches,
                "future_work_evidence": evidence
            })
            for q in unique_matches:
                all_quotes.append(f"论文《{title}》提到：{q}")

    summary = "未配置模型 API，无法生成综合洞察。"
    if llm_client and all_quotes:
        prompt = f"""你是一个敏锐的科研导师。下面是从多篇文献中提取到的关于“局限性”或“未来工作”的原文片段。
请仔细阅读这些片段，完成以下任务：
1. 考虑到论文的演进时间（如 Transformer -> BERT/ViT -> GPT-3），剔除已经被后续论文解决或部分解决的旧 idea。
2. 提炼出 3 个真正具有前瞻性、未被解决的跨领域科研方向。
3. 每个方向必须附上一句简短的理由。

提取的原文片段如下：
{chr(10).join(all_quotes[:20])}

输出格式要求：直接输出 Markdown 列表，不要多余废话。
"""
        try:
            summary = llm_text(prompt, system="你是资深学术导师，擅长洞察技术演进趋势。", temperature=0.3)
        except Exception as exc:
            summary = f"大模型生成失败: {exc}"

    return {
        "items": items,
        "summary": summary
    }


@app.post("/api/futurework")
def get_futurework(user: dict = Depends(current_user)):
    return _future_work(user["id"])


def _arxiv_search(query: str, max_results: int = 8, sort_by: str = 'relevance') -> list[dict]:
    import httpx
    if not query.strip(): query = "cat:cs.AI"
    url = "https://export.arxiv.org/api/query"
    search_query = query if re.search(r"\b(?:all|ti|au|abs|cat|co|jr|rn):", query, re.I) else f"all:{query}"

    # 增加 User-Agent 伪装，并在遇到 429 时重试一次
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    params = {"search_query": search_query, "start": 0, "max_results": max_results, "sortBy": sort_by, "sortOrder": "descending"}

    try:
        response = httpx.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            import time
            time.sleep(3) # 被限流后等 3 秒重试
            response = httpx.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
        else:
            raise
    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        paper_url = entry.findtext("a:id", default="", namespaces=ns)
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        summary = " ".join(entry.findtext("a:summary", default="", namespaces=ns).split())
        published = entry.findtext("a:published", default="", namespaces=ns)
        authors = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
        results.append({"id": paper_url.rstrip("/").split("/")[-1], "title": title, "summary": summary, "published": published, "authors": authors, "url": paper_url, "pdf_url": paper_url.replace("/abs/", "/pdf/") + ".pdf"})
    return results


@app.get("/api/arxiv/search")
def arxiv_search(q: str = Query(..., min_length=2), limit: int = Query(8, ge=1, le=30), user: dict = Depends(current_user)):
    try:
        return {"items": _arxiv_search(q, limit)}
    except Exception as exc:
        raise HTTPException(502, f"arXiv 检索失败：{exc}")


@app.post("/api/arxiv/download")
def download_arxiv(payload: dict, user: dict = Depends(current_user)):
    import httpx
    arxiv_id = str(payload.get("arxiv_id", "")).strip()
    if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv_id):
        raise HTTPException(400, "请提供合法的 arXiv ID，例如 1706.03762v7")
    url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    try:
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        result = _upload_bytes(user, f"{arxiv_id}.pdf", response.content)
        return {"source_url": url, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"arXiv PDF 下载失败：{exc}")


@app.get("/api/recommendations")
def recommendations(user: dict = Depends(current_user), limit: int = Query(8, ge=1, le=20)):
    library = _paper_rows(user["id"])
    if not library:
        return {"items": [], "reason": "请先上传文献，系统才能学习研究兴趣。"}
    topics = " ".join((p.get("title") or p["filename"]).replace(".pdf", "") for p in library[:5])
    # Use a small query from the user's own library; no claim of complete coverage.
    words = [t for t in re.findall(r"[A-Za-z]{3,}", topics) if t.lower() not in {"paper", "arxiv", "version"}]
    query = " ".join(words[:8]) or "machine learning"
    # 优先过滤包含代码仓库的论文 (模拟逻辑)
    try:
        items = _arxiv_search(query, limit)
        return {"items": items, "query": query, "reason": "根据本地文献标题构造主题检索；请人工确认相关性。"}
    except Exception as exc:
        return {"items": [], "query": query, "warning": str(exc)}


# -----------------------------------------------------------------------------
# Translation, academic writing, CSV plotting
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _cached_query_translation(query: str) -> str:
    """Translate Chinese research queries for Reranker scoring."""
    prompt = (
        "\u8bf7\u5c06\u4ee5\u4e0b\u4e2d\u6587\u79d1\u7814\u95ee\u9898\u51c6\u786e\u7ffb\u8bd1\u4e3a\u82f1\u6587\u3002"
        "\u4fdd\u7559\u4e13\u4e1a\u672f\u8bed\u3001\u53d8\u91cf\u540d\u548c\u4e13\u6709\u540d\u8bcd\u3002"
        "\u53ea\u8f93\u51fa\u82f1\u6587\u8bd1\u6587\uff0c\u4e0d\u89e3\u91ca\uff0c\u4e0d\u56de\u7b54\u95ee\u9898\u3002\n\n"
        + query[:2000]
    )

    result = llm_text(
        prompt,
        system="\u4f60\u662f\u4e13\u4e1a\u7684\u4e2d\u82f1\u6587\u5b66\u672f\u7ffb\u8bd1\u52a9\u624b\u3002",
        temperature=0,
    )

    translated = str(result or "").strip()

    if not translated:
        raise ValueError("translation result is empty")

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", translated))
    if chinese_chars > len(translated) * 0.1:
        raise ValueError("translation contains too many Chinese characters")

    if len(translated) > max(500, len(query) * 8):
        raise ValueError("translation result is unexpectedly long")

    return translated

def _build_bilingual_rerank_query(query: str) -> str:
    """???? Reader Reranker ??????????"""
    query = str(query or "").strip()
    if not query or not llm_client:
        return query

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", query))
    if chinese_chars <= len(query) * 0.1:
        return query

    try:
        english = _cached_query_translation(query)
        log.info(
            "[RERANK_QUERY] bilingual query enabled; original_chars=%d translated_chars=%d",
            len(query),
            len(english),
        )
        return query + "\n" + english
    except Exception as exc:
        log.warning("[RERANK_QUERY] translation failed; fallback to original query: %s", exc)
        return query


@app.post("/api/translate")
def translate_text(payload: dict, user: dict = Depends(current_user)):
    text = str(payload.get("text", ""))
    if not text.strip(): raise HTTPException(400, "待翻译文本不能为空")
    if not llm_client: raise HTTPException(503, "尚未配置模型 API")

    # 自动检测语种：如果中文字符占比大于 10%，认为是中文，翻译成英文；否则翻译成中文。
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    is_chinese = chinese_chars > len(text) * 0.1

    if is_chinese:
        prompt = "将以下中文学术文本准确翻译为英文。保留公式、LaTeX、代码、引用编号、变量名和专有名词。不要增删原文事实。\n\n" + text[:12000]
        sys_prompt = "你是专业学术论文译者，将中文翻译为英文。"
    else:
        prompt = "将以下英文学术文本准确翻译为简体中文。保留公式、LaTeX、代码、引用编号、变量名和专有名词。不要增删原文事实。\n\n" + text[:12000]
        sys_prompt = "你是专业学术论文译者，将英文翻译为简体中文。"

    result = llm_text(
        prompt,
        system=sys_prompt,
        temperature=0,
    )
    return {"translation": result, "target_language": "en" if is_chinese else "zh"}


@app.post("/api/write/framework")
def write_framework(payload: dict, user: dict = Depends(current_user)):
    idea = str(payload.get("idea", "")).strip()
    if not idea: raise HTTPException(400, "科研 Idea 不能为空")
    sources = retrieve(user["id"], idea, top_k=10)
    if not llm_client:
        raise HTTPException(503, "尚未配置模型 API")
    prompt = f"""围绕研究想法生成可供研究者继续修改的论文框架：{idea}
结构必须包含 Abstract、Introduction、Related Work、Research Questions、Methodology、Experiments、Limitations。
规则：
1. 只能引用下方证据中真实出现的文献，使用 [E1] 这样的标记。
2. 证据不足时写 [需要补充真实文献]，禁止虚构作者、年份、标题、DOI 或实验结果。
3. 明确区分已有文献结论与拟开展的研究计划。
4. 输出后附“证据与引用清单”，注明论文文件名、页码和原文依据。

可用文献证据：
{_source_context(sources) if sources else '当前知识库未检索到相关文献。请勿编造 References。'}
"""
    framework = llm_text(prompt, system="你是严谨的学术写作助手，优先保证引用真实性。", temperature=0.2)
    return {"framework": framework, "sources": sources, "citation_warning": "请在投稿前逐条核验原文和参考文献格式。"}


@app.post("/api/csv/plot")
async def csv_plot(file: UploadFile = File(...), user: dict = Depends(current_user)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "仅支持 CSV 文件")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024: raise HTTPException(413, "CSV 最大支持 15 MB")
    try:
        import io
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty or len(df.columns) < 2: raise ValueError("CSV 至少需要两列且不能是空文件")
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric: raise ValueError("未找到数值列，请检查 CSV 数据")
        x_col = df.columns[0]
        y_col = numeric[0]
        plot_id = uuid.uuid4().hex[:14]
        output = MEDIA_DIR / f"chart_{plot_id}.png"
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
        if pd.api.types.is_numeric_dtype(df[x_col]) and len(numeric) >= 2:
            ax.scatter(df[x_col], df[numeric[1]], alpha=0.75)
            ax.set_xlabel(str(x_col)); ax.set_ylabel(str(numeric[1]))
            chosen_y = numeric[1]
        else:
            chart_df = df[[x_col, y_col]].dropna().head(80)
            ax.plot(chart_df[x_col].astype(str), chart_df[y_col], marker="o", linewidth=2)
            ax.tick_params(axis="x", rotation=35)
            ax.set_xlabel(str(x_col)); ax.set_ylabel(str(y_col))
            chosen_y = y_col
        ax.grid(alpha=0.25)
        ax.set_title("Experimental Results")
        fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)
        with db() as conn:
            conn.execute(
                "INSERT INTO media_assets(filename, user_id, created_at) VALUES(?,?,?)",
                (output.name, user["id"], _iso(_now())),
            )
        stats_text = df[[x_col, chosen_y]].describe().to_string()
        if llm_client:
            caption = llm_text(
                f"根据下列数据统计摘要生成一条英文 academic Figure Caption，并用中文另附一句客观趋势解读。不要推断因果。\n{stats_text}",
                system="你是学术论文图表编辑。",
                temperature=0.1,
            )
        else:
            caption = f"Figure: {chosen_y} across {x_col}. The plot summarizes {len(df)} observations. 请配置模型 API 以生成更完整的学术图注。"
        return {"image_url": f"/api/media/{output.name}", "caption": caption, "columns": list(df.columns), "rows": len(df), "x_column": str(x_col), "y_column": str(chosen_y)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"CSV 解析或绘图失败：{exc}")


# -----------------------------------------------------------------------------
# Literature review: evidence-first synthesis over the user's paper collection.
# -----------------------------------------------------------------------------
def _generate_literature_review(user_id: str, topic: str, paper_ids: Optional[list[str]] = None) -> dict:
    _review_started = time.perf_counter()
    _review_retrieval_started = time.perf_counter()
    sources = retrieve(user_id, topic, paper_ids=paper_ids, top_k=18)
    log.info(
        "[PERF][REVIEW] retrieval=%.2fs sources=%d",
        time.perf_counter() - _review_retrieval_started,
        len(sources),
    )
    if not sources:
        report = "当前文献库没有检索到足够证据。请先上传相关论文，或调整综述主题。"
    elif llm_client:
        prompt = f"""请基于所给论文片段撰写一份有证据约束的主题文献综述。
综述主题：{topic}
请组织为：1. 研究背景与问题定义；2. 主要技术路线/主题聚类；3. 方法演进；4. 代表性结果与差异；5. 共同局限；6. 可进一步验证的研究方向。
规则：每个事实性判断必须使用 [E编号] 标记；只能使用提供的证据；区分原文事实和综合推断；不能根据未检索到的内容断言不存在相关研究；不虚构作者、年份、DOI 或实验数据。
证据：
{_source_context(sources)}
"""
        report = llm_text(prompt, system="你是谨慎的系统性文献综述研究助理。证据覆盖有限时必须声明局限。", temperature=0.15)
    else:
        report = "尚未配置大模型 API，已检索到以下证据片段，可据此人工整理综述：\n" + _source_context(sources)
    log.info(
        "[PERF][REVIEW] total=%.2fs sources=%d",
        time.perf_counter() - _review_started,
        len(sources),
    )
    return {"topic": topic, "report": report, "sources": sources, "coverage_warning": "本报告基于当前文献库的检索结果，并非覆盖全部数据库的系统综述。"}


@app.post("/api/research/review")
def research_review(payload: dict, user: dict = Depends(current_user)):
    topic = str(payload.get("topic", "")).strip()
    paper_ids = [str(x) for x in payload.get("paper_ids", [])][:12]
    if len(topic) < 3:
        raise HTTPException(400, "请输入至少 3 个字符的综述主题")
    return _generate_literature_review(user["id"], topic, paper_ids or None)


# -----------------------------------------------------------------------------
# Research-gap validation: local evidence + recent ArXiv attempts + cautious synthesis
# -----------------------------------------------------------------------------
def _validate_research_gap(user_id: str, idea: str) -> dict:
    local_sources = retrieve(user_id, idea, top_k=8)
    terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", idea) if w.lower() not in {"with", "from", "that", "this", "using", "based", "into"}]
    query = " ".join(terms[:7]) or "machine learning"
    recent: list[dict] = []
    arxiv_warning = ""
    try:
        recent = _arxiv_search(query, 8)
    except Exception as exc:
        arxiv_warning = str(exc)
    if llm_client:
        prompt = f"""评估以下科研想法是否存在值得探索的研究空间。不要宣称已经证明“全球首创”。
科研想法：{idea}
本地文献证据：
{_source_context(local_sources) if local_sources else '本地库没有检索到相关证据。'}
近期 arXiv 检索结果（检索覆盖不完整）：
{json.dumps([{k:x.get(k) for k in ('id','title','published','summary','url')} for x in recent], ensure_ascii=False)[:14000]}
请输出：
1. 研究问题拆解；2. 已有工作触及了什么；3. 可能的研究空白（明确标注为假设）；4. 最接近的 3 项工作及差异；5. 下一步验证实验；6. 新颖性风险与检索局限。
每个本地论文事实必须标注 [E编号]；arXiv 结果标注 arXiv ID 和链接。若搜索结果已出现相似方向，必须明确提示。不得捏造实验结果或断言绝对新颖。
"""
        analysis = llm_text(prompt, system="你是谨慎的科研选题评审专家。证据优先，禁止夸大创新性。", temperature=0.1)
    else:
        analysis = "当前未配置大模型，已完成本地证据检索和近期 arXiv 检索。请配置 LLM_API_KEY 后生成研究空白评估报告。"
    return {"idea": idea, "analysis": analysis, "local_evidence": local_sources, "recent_arxiv": recent, "arxiv_warning": arxiv_warning, "disclaimer": "该结果是辅助筛查，不构成全面文献系统综述或绝对新颖性证明。"}


@app.post("/api/research/validate-idea")
def validate_idea(payload: dict, user: dict = Depends(current_user)):
    idea = str(payload.get("idea", "")).strip()
    if len(idea) < 8:
        raise HTTPException(400, "请提供更完整的科研 Idea（至少 8 个字符）")
    return _validate_research_gap(user["id"], idea)


# -----------------------------------------------------------------------------
# Agent planner, tool registry and deterministic workflow execution
# -----------------------------------------------------------------------------
TOOL_DESCRIPTIONS = {
    "library_stats": "统计当前用户的文献数量、页数和索引块数",
    "search_library": "按自然语言问题检索用户本地论文并返回页码证据",
    "compare_library": "跨多篇论文独立检索证据并形成对比",
    "extract_future_work": "提取当前文献库中论文的 Future Work 与 Limitations",
    "build_citation_graph": "从 DOI/论文标题线索构建本地引用网络并计算 PageRank",
    "generate_framework": "根据科研 Idea 和真实检索证据生成论文框架",
    "generate_literature_review": "根据文献库证据生成带引用标记的主题综述",
    "search_arxiv": "检索 arXiv 最新相关论文",
    "download_arxiv_paper": "根据 arXiv ID 从 arXiv 官方 PDF 地址下载论文并入库",
    "validate_research_gap": "结合本地论文证据与近期 arXiv 工作，审慎评估研究空白和创新风险",
}


# -----------------------------------------------------------------------------
# Agent tool argument schemas
# -----------------------------------------------------------------------------

# 统一的参数字符串规则。
ToolQuery = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]

ToolIdea = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=8,
        max_length=2000,
    ),
]

ToolReviewTopic = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=3,
        max_length=300,
    ),
]

ToolArxivQuery = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=300,
    ),
]

ToolPaperId = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=64,
    ),
]

ToolArxivId = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        pattern=r"^\d{4}\.\d{4,5}(?:v\d+)?$",
    ),
]


class ToolArgsBase(BaseModel):
    # 不允许模型随意添加工具没有定义的参数。
    model_config = {"extra": "forbid"}


class NoToolArgs(ToolArgsBase):
    pass


class SearchLibraryArgs(ToolArgsBase):
    query: ToolQuery


class CompareLibraryArgs(ToolArgsBase):
    query: ToolQuery
    paper_ids: list[ToolPaperId] = Field(
        default_factory=list,
        max_length=8,
    )


class GenerateFrameworkArgs(ToolArgsBase):
    idea: ToolIdea


class GenerateLiteratureReviewArgs(ToolArgsBase):
    topic: ToolReviewTopic


class SearchArxivArgs(ToolArgsBase):
    query: ToolArxivQuery = "machine learning"
    sort_by: Literal["relevance", "latest"] = "relevance"


class DownloadArxivPaperArgs(ToolArgsBase):
    arxiv_id: ToolArxivId


class ValidateResearchGapArgs(ToolArgsBase):
    idea: ToolIdea


# 每个工具对应自己的参数模型。
TOOL_ARG_SCHEMAS = {
    "library_stats": NoToolArgs,
    "search_library": SearchLibraryArgs,
    "compare_library": CompareLibraryArgs,
    "extract_future_work": NoToolArgs,
    "build_citation_graph": NoToolArgs,
    "generate_framework": GenerateFrameworkArgs,
    "generate_literature_review": GenerateLiteratureReviewArgs,
    "search_arxiv": SearchArxivArgs,
    "download_arxiv_paper": DownloadArxivPaperArgs,
    "validate_research_gap": ValidateResearchGapArgs,
}


def _planner(user_input: str, paper_ids: list[str]) -> dict:
    fallback = {"goal": user_input, "steps": []}
    text = user_input.lower()

    # Search-then-download is a multi-step task, not a direct download.
    is_arxiv_download = (
        ("\u4e0b\u8f7d" in text or "download" in text)
        and "arxiv" in text
    )
    has_search_intent = any(
        word in text
        for word in [
            "\u641c\u7d22",
            "\u68c0\u7d22",
            "\u67e5\u627e",
            "search",
            "find",
        ]
    )
    if is_arxiv_download and has_search_intent:
        latest_intent = any(
            word in text
            for word in [
                "\u6700\u65b0",
                "\u8fd1\u671f",
                "\u6700\u8fd1",
                "\u8fd1\u4e00\u5e74",
                "\u8fc7\u53bb\u4e00\u5e74",
                "latest",
                "newest",
                "most recent",
                "past year",
                "last year",
            ]
        )
        fallback["steps"] = [
            {
                "tool": "search_arxiv",
                "args": {
                    "query": user_input,
                    "sort_by": "latest" if latest_intent else "relevance",
                },
            },
            {
                "tool": "download_arxiv_paper",
                "args": {
                    "arxiv_id": {
                        "$from_step": 1,
                        "path": "items.0.id",
                    }
                },
            },
        ]
        return fallback

    # Explicit direct arXiv download requests take priority over LLM planning.
    is_arxiv_download = (
        ("\u4e0b\u8f7d" in text or "download" in text)
        and "arxiv" in text
    )
    if is_arxiv_download:
        arxiv_match = ARXIV_RE.search(user_input)
        if arxiv_match:
            fallback["steps"] = [{
                "tool": "download_arxiv_paper",
                "args": {"arxiv_id": arxiv_match.group(0)}
            }]
        else:
            fallback["clarification"] = (
                "\u8bf7\u63d0\u4f9b\u9700\u8981\u4e0b\u8f7d\u7684 arXiv ID\uff0c"
                "\u4f8b\u5982 2301.12345 \u6216 2301.12345v2\u3002"
            )
        return fallback

    if any(x in text for x in ["多少篇", "数量", "统计", "文献库有几"]):
        fallback["steps"] = [{"tool": "library_stats", "args": {}}]
    elif any(x in text for x in ["对比", "比较", "演进", "差异"]):
        fallback["steps"] = [{"tool": "compare_library", "args": {"paper_ids": paper_ids, "query": user_input}}]
    elif any(x in text for x in ["综述", "literature review", "systematic review"]):
        fallback["steps"] = [{"tool": "generate_literature_review", "args": {"topic": user_input}}]
    elif any(x in text for x in ["创新点", "创新性", "idea 验证", "研究空白", "新颖性", "有没有人做过"]):
        fallback["steps"] = [{"tool": "validate_research_gap", "args": {"idea": user_input}}]
    elif any(x in text for x in ["future work", "未来工作", "研究方向"]):
        fallback["steps"] = [{"tool": "extract_future_work", "args": {}}]
    elif any(x in text for x in ["图谱", "引用网络", "基石论文", "pagerank"]):
        fallback["steps"] = [{"tool": "build_citation_graph", "args": {}}]
    elif "下载" in text and "arxiv" in text:
        arxiv_match = ARXIV_RE.search(user_input)
        if arxiv_match:
            fallback["steps"] = [{
                "tool": "download_arxiv_paper",
                "args": {"arxiv_id": arxiv_match.group(0)}
            }]
        else:
            fallback["clarification"] = (
                "\u8bf7\u63d0\u4f9b\u9700\u8981\u4e0b\u8f7d\u7684 arXiv ID\uff0c\u4f8b\u5982 2301.12345 \u6216 2301.12345v2\u3002"
            )
    elif any(x in text for x in ["arxiv", "最新论文", "新论文", "推荐论文"]):
        fallback["steps"] = [{"tool": "search_arxiv", "args": {"query": user_input}}]
    elif any(x in text for x in ["框架", "abstract", "idea", "综述", "related work"]):
        fallback["steps"] = [
            {"tool": "search_library", "args": {"query": user_input}},
            {"tool": "generate_framework", "args": {"idea": user_input}},
        ]
    else:
        fallback["steps"] = [{"tool": "search_library", "args": {"query": user_input}}]
    if not llm_client:
        return fallback
    try:
        tool_contracts = {
            name: {
                "description": TOOL_DESCRIPTIONS[name],
                "arguments_schema": schema.model_json_schema(),
            }
            for name, schema in TOOL_ARG_SCHEMAS.items()
        }

        prompt = f"""User goal: {user_input}
Selected paper IDs: {paper_ids}

Available tools and their actual argument schemas:
{json.dumps(tool_contracts, ensure_ascii=False)}

Return 1-4 steps as JSON:
{{"goal":"...","steps":[{{"tool":"tool_name","args":{{}}}}]}}

STRICT RULES:
1. Only use tools listed above.
2. Arguments must follow the corresponding arguments_schema.
3. Never invent unsupported argument fields.
4. Required fields must be supplied. No-argument tools use an empty object.
5. compare_library paper_ids must come from the selected paper IDs.
6. generate_literature_review only accepts topic and uses local library evidence.
7. search_arxiv accepts query and sort_by. Use sort_by="latest" when the user asks for the newest or most recent papers; otherwise use "relevance".
8. download_arxiv_paper requires a valid arXiv ID, or a restricted reference to the immediately previous search_arxiv step: {{"$from_step": 1, "path": "items.0.id"}}.
9. Prefer evidence retrieval before analysis.
10. Never call arbitrary code or unlisted tools.
"""
        plan = llm_json(prompt, system="你是科研任务规划 Agent。你必须严格输出工具白名单内的 JSON 计划。")
        steps = plan.get("steps", [])
        safe_steps = []
        clarification = None
        for step_index, step in enumerate(steps[:4], start=1):
            if isinstance(step, dict) and step.get("tool") in TOOL_DESCRIPTIONS:
                tool_name = step["tool"]
                tool_args = step.get("args") if isinstance(step.get("args"), dict) else {}

                # 核心修复：如果大模型规划了工具但忘记传核心参数，强制用用户的原始提问兜底
                if tool_name == "validate_research_gap" and not tool_args.get("idea"):
                    tool_args["idea"] = user_input
                elif tool_name == "generate_framework" and not tool_args.get("idea"):
                    tool_args["idea"] = user_input
                elif tool_name == "search_library" and not tool_args.get("query"):
                    tool_args["query"] = user_input
                elif tool_name == "generate_literature_review" and not tool_args.get("topic"):
                    tool_args["topic"] = user_input

                if tool_name == "download_arxiv_paper":
                    candidate_value = tool_args.get("arxiv_id", "")

                    if isinstance(candidate_value, dict):
                        reference_path = candidate_value.get("path")
                        valid_reference = (
                            set(candidate_value.keys()) == {"$from_step", "path"}
                            and isinstance(candidate_value.get("$from_step"), int)
                            and not isinstance(candidate_value.get("$from_step"), bool)
                            and candidate_value.get("$from_step") == step_index - 1
                            and isinstance(reference_path, str)
                            and bool(re.fullmatch(
                                r"items\.(0|[1-9]\d?)\.id",
                                reference_path,
                            ))
                        )

                        previous_step = (
                            steps[step_index - 2]
                            if step_index >= 2 and step_index - 2 < len(steps)
                            else {}
                        )

                        if (
                            not valid_reference
                            or not isinstance(previous_step, dict)
                            or previous_step.get("tool") != "search_arxiv"
                        ):
                            clarification = "无法解析 arXiv 论文引用，请检查引用路径是否为 items.0.id 至 items.99.id，并确认前一步是 arXiv 搜索。"
                            break
                    else:
                        candidate_id = str(candidate_value).strip()
                        if not ARXIV_RE.fullmatch(candidate_id):
                            clarification = "\u8bf7\u63d0\u4f9b\u5408\u6cd5\u7684 arXiv ID\uff0c\u4f8b\u5982 2301.12345 \u6216 2301.12345v2\u3002"
                            break
                if tool_name == "search_arxiv":
                    if not tool_args.get("query"):
                        tool_args["query"] = user_input

                    latest_intent = any(
                        word in text
                        for word in [
                            "\u6700\u65b0",
                            "\u8fd1\u671f",
                            "\u6700\u8fd1",
                            "latest",
                            "newest",
                            "most recent",
                        ]
                    )
                    # ????????????LLM?????????
                    tool_args["sort_by"] = (
                        "latest" if latest_intent else "relevance"
                    )

                safe_steps.append({"tool": tool_name, "args": tool_args})
        if clarification:
            return {
                "goal": user_input,
                "steps": [],
                "clarification": clarification,
            }
        if safe_steps:
            return {"goal": str(plan.get("goal", user_input)), "steps": safe_steps}
    except Exception as exc:
        log.warning("Planner fallback: %s", exc)
    return fallback


def _tool_library_stats(user_id: str) -> dict:
    with db() as conn:
        paper_count = conn.execute("SELECT COUNT(*) FROM papers WHERE user_id=?", (user_id,)).fetchone()[0]
        page_count = conn.execute("SELECT COALESCE(SUM(page_count),0) FROM papers WHERE user_id=?", (user_id,)).fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE p.user_id=?", (user_id,)).fetchone()[0]
    return {"paper_count": paper_count, "page_count": page_count, "chunk_count": chunk_count}


def _run_tool(user_id: str, tool: str, args: dict) -> dict:

    # 统一检查工具名称和参数类型。
    schema = TOOL_ARG_SCHEMAS.get(tool)

    if schema is None:
        raise HTTPException(400, "不允许的 Agent 工具")

    if not isinstance(args, dict):
        raise HTTPException(422, "Agent 工具参数必须是 JSON 对象")

    # 使用对应的 Pydantic 模型验证参数。
    try:
        # 严格校验：拒绝所有未在工具参数模型中定义的字段。
        validated_args = schema.model_validate(args)
        args = validated_args.model_dump()

    except ValidationError as exc:
        errors = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in exc.errors()
        ]

        log.warning(
            "Agent tool argument validation failed: tool=%s errors=%s",
            tool,
            errors,
        )

        raise HTTPException(
            status_code=422,
            detail={
                "message": "Agent 工具参数验证失败",
                "errors": errors,
            },
        ) from exc

    # 对跨论文比较进行用户归属检查，防止访问其他用户的论文。
    if tool == "compare_library":
        for paper_id in args.get("paper_ids", []):
            _get_paper(user_id, paper_id)

    if tool == "library_stats":
        return _tool_library_stats(user_id)
    if tool == "search_library":
        query = str(args.get("query", ""))
        return {"sources": retrieve(user_id, query, top_k=6)}
    if tool == "compare_library":
        ids = [str(x) for x in args.get("paper_ids", []) if x]
        if not ids:
            ids = [p["id"] for p in _paper_rows(user_id)[:4]]
        result = []
        for pid in ids[:8]:
            hits = retrieve(user_id, str(args.get("query", "")), [pid], top_k=3)
            for hit in hits:
                hit["evidence_id"] = f"E{len(result) + 1}"
                result.append(hit)
        return {"sources": result, "paper_ids": ids[:8]}
    if tool == "extract_future_work":
        return _future_work(user_id)
    if tool == "build_citation_graph":
        return _build_graph_data(user_id)
    if tool == "generate_framework":
        idea = str(args.get("idea", ""))
        sources = retrieve(user_id, idea, top_k=8)
        if llm_client:
            framework = llm_text(
                f"为研究想法生成 Abstract/Introduction/Related Work/Methodology/Experiments 框架。引用只能标注下列证据 ID；无证据不得编造。\nIdea：{idea}\n证据：\n{_source_context(sources)}",
                system="你是审慎的论文写作 Agent。",
                temperature=0.2,
            )
        else:
            framework = "尚未配置模型 API。已检索到证据：\n" + _source_context(sources)
        return {"framework": framework, "sources": sources}
    if tool == "generate_literature_review":
        topic = str(args.get("topic", "科研主题综述"))
        return _generate_literature_review(user_id, topic)
    if tool == "search_arxiv":
        query = str(args.get("query", "machine learning"))
        words = re.findall(r"[A-Za-z0-9:._-]+", query)
        clean_query = " ".join(words[:8]) or "machine learning"
        sort_by = args.get("sort_by", "relevance")

        if sort_by == "latest":
            # ?????????30????
            stop_words = {
                "the", "and", "for", "with", "from",
                "about", "paper", "papers", "latest",
                "recent", "new", "arxiv",
            }
            query_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9]+", clean_query)
                if len(token) >= 3 and token.lower() not in stop_words
            }
            is_rag_query = (
                "rag" in query_terms
                and "retrieval" in query_terms
                and "generation" in query_terms
            )

            if is_rag_query:
                # ??/????????????? arXiv ID ???
                rag_queries = [
                    'ti:"retrieval-augmented generation" OR abs:"retrieval-augmented generation"',
                    'ti:"retrieval augmented generation" OR abs:"retrieval augmented generation"',
                    'ti:RAG AND (abs:retrieval OR abs:augmented OR abs:generation)',
                ]
                candidates_by_id = {}
                for scoped_query in rag_queries:
                    for item in _arxiv_search(
                        scoped_query, 100, "submittedDate"
                    ):
                        paper_id = str(item.get("id", "")).strip()
                        if paper_id:
                            candidates_by_id[paper_id] = item
                candidates = list(candidates_by_id.values())
            else:
                candidates = _arxiv_search(
                    clean_query, 100, "submittedDate"
                )

            relevant = []
            is_rag_query = (
                "rag" in query_terms
                and "retrieval" in query_terms
                and "generation" in query_terms
            )

            def rag_relevance_score(title, summary):
                title = str(title or "").lower()
                summary = str(summary or "").lower()

                phrase = r"\bretrieval[\s-]+augmented[\s-]+generation\b"

                # ?????? RAG?????
                if re.search(phrase, title):
                    return 10

                if re.search(r"\brag\b", title):
                    return 9

                if (
                    re.search(r"\bretrieval\b", title)
                    and re.search(r"\b(augmented|generation)\b", title)
                ):
                    return 8

                sentences = re.split(r"(?<=[.!?])\s+", summary)

                action_pattern = (
                    r"\b(propose|proposes|proposed|present|presents|"
                    r"presented|introduce|introduces|introduced|"
                    r"develop|develops|developed|design|designs|"
                    r"designed|build|builds|built|evaluate|evaluates|"
                    r"evaluated)\b"
                )

                # ??????????????????
                comparison_pattern = (
                    r"\b(compared\s+(?:with|to)|versus|vs\.?|"
                    r"baseline|baselines)\b"
                )

                # RAG-assisted / RAG-based ?????????????
                component_pattern = (
                    phrase
                    + r"\s*(?:\(\s*rag\s*\))?\s*[-?]\s*"
                    + r"(?:assisted|based|powered|enabled)\b"
                )

                # ?????????? RAG?????????????
                core_relation_pattern = (
                    r"\b(?:for|in|to|of)\s+" + phrase
                )

                for sentence in sentences:
                    if not re.search(phrase, sentence):
                        continue

                    if re.search(comparison_pattern, sentence):
                        continue

                    if re.search(component_pattern, sentence):
                        continue

                    if not re.search(core_relation_pattern, sentence):
                        continue

                    if re.search(action_pattern, sentence):
                        return 7

                return 0

            relevant = []

            for item in candidates:
                title = str(item.get("title", ""))
                summary = str(item.get("summary", ""))

                if is_rag_query:
                    score = rag_relevance_score(title, summary)
                    is_relevant = score >= 6
                else:
                    searchable = (
                        title.lower() + " " + summary.lower()
                    )
                    matched_terms = {
                        term for term in query_terms
                        if re.search(
                            r"\b" + re.escape(term) + r"\b",
                            searchable,
                        )
                    }
                    minimum_matches = min(2, len(query_terms))
                    is_relevant = (
                        not query_terms
                        or len(matched_terms) >= minimum_matches
                    )

                if is_relevant:
                    relevant.append(item)

            # arXiv published ??? ISO ?????
            def published_timestamp(item):
                value = str(item.get("published", "")).strip()
                try:
                    return datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    ).timestamp()
                except (ValueError, TypeError, OverflowError):
                    return float("-inf")

            relevant.sort(
                key=published_timestamp,
                reverse=True,
            )

            return {
                "items": relevant[:5],
                "sort_by": "latest",
                "candidate_count": len(candidates),
                "filtered_count": len(relevant),
            }

        # ???????????????6????
        return {
            "items": _arxiv_search(clean_query, 6),
            "sort_by": "relevance",
        }
    if tool == "validate_research_gap":
        return _validate_research_gap(user_id, str(args.get("idea", "")))
    if tool == "download_arxiv_paper":
        import httpx
        arxiv_id = str(args.get("arxiv_id", "")).strip()
        if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv_id):
            return {"error": "未从请求中识别到合法 arXiv ID，请提供如 1706.03762v7 的编号。"}
        url = f"https://export.arxiv.org/pdf/{arxiv_id}"
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        return _upload_bytes({"id": user_id}, f"{arxiv_id}.pdf", response.content)
    raise HTTPException(400, "不允许的工具")


def _resolve_agent_step_args(
    tool: str,
    args: dict,
    observations: list[dict],
    current_step: int,
) -> dict:
    """???????? arXiv ?????? items.N.id?"""
    if not isinstance(args, dict):
        raise HTTPException(422, "Agent 工具参数必须是 JSON 对象")

    resolved = dict(args)

    references = [
        (key, value)
        for key, value in resolved.items()
        if isinstance(value, dict) and "$from_step" in value
    ]

    if not references:
        return resolved

    if tool != "download_arxiv_paper" or len(references) != 1:
        raise HTTPException(422, "当前工具不支持此 arXiv 引用方式")

    key, reference = references[0]

    if key != "arxiv_id" or set(reference.keys()) != {"$from_step", "path"}:
        raise HTTPException(422, "arXiv 引用参数格式不正确")

    source_step = reference.get("$from_step")

    if (
        not isinstance(source_step, int)
        or isinstance(source_step, bool)
        or source_step != current_step - 1
        or source_step < 1
        or source_step > len(observations)
    ):
        raise HTTPException(422, "引用必须指向紧邻的上一步结果")

    reference_path = reference.get("path")

    match = (
        re.fullmatch(r"items\.(0|[1-9]\d?)\.id", reference_path)
        if isinstance(reference_path, str)
        else None
    )

    if not match:
        raise HTTPException(422, "引用路径必须是 items.0.id 至 items.99.id")

    observation = observations[source_step - 1]

    if observation.get("tool") != "search_arxiv" or "result" not in observation:
        raise HTTPException(422, "引用来源必须是 arXiv 搜索结果")

    result = observation["result"]
    items = result.get("items") if isinstance(result, dict) else None

    if not isinstance(items, list) or not items:
        raise HTTPException(422, "arXiv 搜索结果中没有可用论文")

    item_index = int(match.group(1))

    if item_index >= len(items) or not isinstance(items[item_index], dict):
        raise HTTPException(422, "指定索引对应的 arXiv 论文不存在")

    arxiv_id = items[item_index].get("id")

    if not isinstance(arxiv_id, str) or not ARXIV_RE.fullmatch(arxiv_id):
        raise HTTPException(422, "论文 ID 缺失或格式不合法")

    resolved[key] = arxiv_id
    return resolved


@app.post("/api/agent/chat")
def agent_chat(payload: dict, user: dict = Depends(current_user)):
    user_input = str(payload.get("message", "")).strip()
    paper_ids = [str(x) for x in payload.get("paper_ids", [])][:12]
    if not user_input: raise HTTPException(400, "请输入科研任务")
    def generate():
        yield _sse({"type": "thought", "content": "任务规划 Agent：正在识别目标、约束和所需证据……"})
        plan = _planner(user_input, paper_ids)
        if not plan.get("steps") and plan.get("clarification"):
            yield _sse({"type": "thought", "content": "\u4efb\u52a1\u6682\u65e0\u6cd5\u6267\u884c\uff1a\u7f3a\u5c11\u5fc5\u8981\u53c2\u6570\u3002"})
            yield _sse({"type": "text", "content": plan["clarification"]})
            return
        yield _sse({"type": "thought", "content": f"任务计划已生成：共 {len(plan['steps'])} 个步骤。"})
        observations = []
        for index, step in enumerate(plan["steps"], start=1):
            tool = step["tool"]
            yield _sse({"type": "action", "content": f"步骤 {index}：调用 {tool} — {TOOL_DESCRIPTIONS[tool]}"})
            try:
                resolved_args = _resolve_agent_step_args(
                    tool,
                    step.get("args", {}),
                    observations,
                    index,
                )
                _tool_started = time.perf_counter()
                result = _run_tool(user["id"], tool, resolved_args)
                log.info(
                    "[PERF][AGENT] tool=%s elapsed=%.2fs",
                    tool,
                    time.perf_counter() - _tool_started,
                )
                observations.append({"tool": tool, "result": result})
                brief = json.dumps(result, ensure_ascii=False)
                yield _sse({"type": "observation", "content": brief[:3500]})
            except Exception as exc:
                observations.append({"tool": tool, "error": str(exc)})
                yield _sse({"type": "error", "content": f"{tool} 执行失败：{exc}"})
        yield _sse({"type": "final", "content": "正在汇总证据与工具结果……"})
        # Return structured tool results directly for simple queries; use the LLM
        # only to synthesize and never silently invent source facts.
        flat_sources = []
        for item in observations:
            res = item.get("result", {})
            if isinstance(res, dict) and isinstance(res.get("sources"), list):
                normalized_sources = []
                for source in res["sources"]:
                    source = dict(source)
                    source["evidence_id"] = f"E{len(flat_sources) + 1}"
                    flat_sources.append(source)
                    normalized_sources.append(source)
                res["sources"] = normalized_sources
        if (
            len(observations) == 1
            and observations[0].get("tool") == "search_arxiv"
            and isinstance(observations[0].get("result"), dict)
        ):
            result = observations[0]["result"]
            items = result.get("items", [])

            if not items:
                yield _sse({
                    "type": "text",
                    "content": "\u6ca1\u6709\u627e\u5230\u7b26\u5408\u4e3b\u9898\u8fc7\u6ee4\u6761\u4ef6\u7684 arXiv \u8bba\u6587\u3002\u53ef\u4ee5\u5c1d\u8bd5\u6269\u5927\u6216\u8c03\u6574\u68c0\u7d22\u5173\u952e\u8bcd\u3002",
                })
                return

            lines = [
                "## arXiv \u68c0\u7d22\u7ed3\u679c",
                "",
                "\u6392\u5e8f\u65b9\u5f0f\uff1a" + ("\u63d0\u4ea4\u65f6\u95f4\u5012\u5e8f" if result.get("sort_by") == "latest" else "\u76f8\u5173\u6027"),
                f"\u5171\u8fd4\u56de {len(items)} \u7bc7\u8bba\u6587\u3002",
                "",
            ]

            for index, item in enumerate(items, start=1):
                lines.extend([
                    f"### {index}. {item.get('title') or 'Untitled'}",
                    f"- arXiv ID\uff1a{item.get('id', 'unknown')}",
                    f"- \u53d1\u5e03\u65f6\u95f4\uff1a{str(item.get('published', ''))[:10]}",
                    f"- \u4f5c\u8005\uff1a{', '.join(item.get('authors', [])) or 'N/A'}",
                    f"- \u94fe\u63a5\uff1a{item.get('url', '')}",
                    "",
                ])

            yield _sse({
                "type": "text",
                "content": "\n".join(lines),
            })
            return

        if llm_client:
            evidence_context = _source_context(flat_sources[:20]) if flat_sources else json.dumps(observations, ensure_ascii=False)[:16000]
            prompt = f"用户目标：{user_input}\n执行计划：{json.dumps(plan, ensure_ascii=False)}\n工具结果：{json.dumps(observations, ensure_ascii=False)[:18000]}\n可引用证据：\n{evidence_context}\n请中文总结完成情况。事实必须带 [E编号]；没有证据的内容明确说明；列出下一步建议。"
            try:
                stream = llm_client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1, stream=True)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta: yield _sse({"type": "text", "content": delta})
            except Exception as exc:
                yield _sse({"type": "text", "content": f"工具已执行，但总结模型调用失败：{exc}\n\n工具结果：{json.dumps(observations, ensure_ascii=False)[:8000]}"})
        else:
            yield _sse({"type": "text", "content": "已完成任务工具调用。当前未配置大模型 API，以下为工具结构化结果：\n" + json.dumps(observations, ensure_ascii=False, indent=2)[:9000]})
    return _stream_response(generate())


@app.get("/api/agent/tools")
def list_agent_tools(user: dict = Depends(current_user)):
    return {"tools": [{"name": key, "description": value} for key, value in TOOL_DESCRIPTIONS.items()]}


# Backwards-compatible simple endpoint for older frontends.
@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return user
