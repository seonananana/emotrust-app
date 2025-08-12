# main.py
# -*- coding: utf-8 -*-

import os
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hashlib import sha256

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

# 분석 파이프라인 & 민팅 헬퍼
from analyzer import pre_pipeline
from mint.mint import send_mint, wait_token_id
from dotenv import load_dotenv; load_dotenv()

BASE = Path(__file__).resolve().parent  # backend/
load_dotenv(BASE / ".env")                                  # backend/.env
load_dotenv(BASE.parent / "hardhat" / ".env", override=False)  # hardhat/.env도 fallback

# ────────────────────────────────────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("emotrust-backend")

APP_VERSION = "1.4.0"
DB_PATH = os.getenv("DB_PATH", "emotrust.db")
USE_DB = os.getenv("USE_DB", "true").lower() == "true"   # false면 파일(JSONL) 저장으로 대체

# ────────────────────────────────────────────────────────────────────────────────
# 선택: 파일(JSONL) 저장 유틸 (USE_DB=false일 때 사용)
# ────────────────────────────────────────────────────────────────────────────────
POSTS_LOG_PATH = os.getenv("POSTS_LOG_PATH", "./data/posts.jsonl")

def _jsonl_append(obj: Dict[str, Any]) -> int:
    os.makedirs(os.path.dirname(POSTS_LOG_PATH), exist_ok=True)
    if "id" not in obj:
        # 밀리초 타임스탬프 기반 ID
        obj["id"] = int(datetime.utcnow().timestamp() * 1000)
    if "created_at" not in obj:
        obj["created_at"] = datetime.utcnow().isoformat() + "Z"
    with open(POSTS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return int(obj["id"])

def _jsonl_read_all() -> List[Dict[str, Any]]:
    if not os.path.exists(POSTS_LOG_PATH):
        return []
    out: List[Dict[str, Any]] = []
    with open(POSTS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _jsonl_get(post_id: int) -> Optional[Dict[str, Any]]:
    for item in reversed(_jsonl_read_all()):
        if int(item.get("id", -1)) == int(post_id):
            return item
    return None

def _jsonl_list(limit: int, offset: int) -> List[Dict[str, Any]]:
    items = list(reversed(_jsonl_read_all()))
    return items[offset: offset + limit]

# ────────────────────────────────────────────────────────────────────────────────
# DB (SQLAlchemy - SQLite)  ※ USE_DB=true일 때만 활성
# ────────────────────────────────────────────────────────────────────────────────
if USE_DB:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
    from sqlalchemy.orm import sessionmaker, declarative_base

    Base = declarative_base()
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    class Post(Base):
        __tablename__ = "posts"
        id = Column(Integer, primary_key=True, autoincrement=True)
        title = Column(Text, nullable=False)
        content = Column(Text, nullable=False)

        # JSON 문자열로 저장(유연성)
        scores_json = Column(Text, nullable=False)      # {S_pre, S_sinc, S_acc, coverage, ...}
        weights_json = Column(Text, nullable=False)     # {w_acc, w_sinc}
        files_json = Column(Text, nullable=False)       # [{name,size}...] or []
        meta_json = Column(Text, nullable=False)        # 프론트·분석 메타

        denom_mode = Column(String(20), default="all")
        gate = Column(Float, default=0.70)
        analysis_id = Column(String(64), index=True, default="")
        created_at = Column(DateTime, default=datetime.utcnow)

    Base.metadata.create_all(engine)
    logger.info(f"🗄️ SQLite ready at {Path(DB_PATH).resolve()}")
else:
    logger.info("🗒️ Running in NO-DB mode (JSONL storage).")

# ────────────────────────────────────────────────────────────────────────────────
# FastAPI + CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="emotrust-backend", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# 스키마 (필수 필드만, 기본값으로 유연하게)
# ────────────────────────────────────────────────────────────────────────────────
class PreResult(BaseModel):
    pii_action: str
    pii_reasons: List[str] = []
    S_acc: float = Field(0.0, ge=0.0, le=1.0)
    S_sinc: float = Field(0.0, ge=0.0, le=1.0)
    S_pre: float = Field(0.0, ge=0.0, le=1.0)
    gate_pass: bool = False
    tokens: int = 0
    matched: int = 0
    total: int = 0
    coverage: float = Field(0.0, ge=0.0, le=1.0)
    clean_text: str = ""
    masked: bool = False
    # 확장 필드(있으면 채우고, 없으면 기본값)
    S_pre_ext: float = Field(0.0, ge=0.0, le=1.0)
    S_fact: Optional[float] = None
    need_evidence: bool = False
    claims: List[str] = []
    evidence: Dict[str, Any] = {}

class AnalyzeResponse(BaseModel):
    ok: bool
    meta: Dict[str, Any]
    result: PreResult

class ScoresIn(BaseModel):
    S_pre: float
    S_sinc: float
    S_acc: Optional[float] = None   # 이름 혼용 대비(S_fact/S_acc)
    S_fact: Optional[float] = None
    coverage: float
    total: int
    matched: int
    masked: bool
    gate_pass: bool

class PostIn(BaseModel):
    title: str
    content: str
    scores: ScoresIn
    weights: Dict[str, float] = {"w_acc": 0.5, "w_sinc": 0.5}
    denom_mode: str = "all"
    gate: float = 0.70
    files: List[Dict[str, Any]] = []
    meta: Optional[Dict[str, Any]] = None
    analysis_id: Optional[str] = None

class PostOut(BaseModel):
    id: int
    title: str
    content: str
    scores: Dict[str, Any]
    weights: Dict[str, Any]
    files: List[Dict[str, Any]]
    meta: Dict[str, Any]
    denom_mode: str
    gate: float
    analysis_id: str
    created_at: str

class AnalyzeMintReq(BaseModel):
    text: str
    comments: int = 0
    to_address: Optional[str] = None
    denom_mode: str = "all"

# ────────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────────
def _await_read_uploadfile(f: UploadFile) -> bytes:
    try:
        return f.file.read()
    finally:
        try:
            f.file.seek(0)
        except Exception:
            pass

def _save_pdfs(pdfs: Optional[List[UploadFile]]) -> List[str]:
    """업로드된 PDF들을 임시 폴더에 저장하고 파일 경로 리스트를 반환."""
    if not pdfs:
        return []
    saved_paths: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="emotrust_pdf_")
    for i, f in enumerate(pdfs):
        name = f.filename or f"evidence_{i}.pdf"
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        dst = Path(tmpdir) / name
        data = _await_read_uploadfile(f)
        with open(dst, "wb") as out:
            out.write(data)
        saved_paths.append(str(dst))
    return saved_paths

def _to_json_str(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), ensure_ascii=False)
        if hasattr(obj, "dict"):
            return json.dumps(obj.dict(), ensure_ascii=False)
    except Exception:
        pass
    return json.dumps(obj, ensure_ascii=False)

def _from_json_str(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

def _call_pre_pipeline_safe(
    text: str, denom_mode: str, w_acc: float, w_sinc: float, gate: float, pdf_paths: Optional[List[str]]
) -> Dict[str, Any]:
    """
    pre_pipeline 시그니처가 버전에 따라 pdf_paths를 안 받을 수도 있어서
    두 방식 모두 시도.
    """
    try:
        return pre_pipeline(text=text, denom_mode=denom_mode, w_acc=w_acc, w_sinc=w_sinc, gate=gate, pdf_paths=pdf_paths)
    except TypeError:
        return pre_pipeline(text=text, denom_mode=denom_mode, w_acc=w_acc, w_sinc=w_sinc, gate=gate)

# ────────────────────────────────────────────────────────────────────────────────
# 라우트
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Hello emotrust"

@app.get("/health")
async def health():
    return {"ok": True, "version": APP_VERSION, "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),      # "all" or "matched"
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: float = Form(0.70),
    pdfs: Optional[List[UploadFile]] = File(None),
):
    try:
        text = f"{title}\n\n{content}".strip() if title else content
        pdf_paths = _save_pdfs(pdfs) if pdfs else []

        out = _call_pre_pipeline_safe(
            text=text, denom_mode=denom_mode, w_acc=w_acc, w_sinc=w_sinc, gate=gate, pdf_paths=pdf_paths
        )

        return AnalyzeResponse(
            ok=True,
            meta={
                "title": title,
                "chars": len(text),
                "pdf_count": len(pdf_paths),
                "pdf_paths": pdf_paths,
                "denom_mode": denom_mode,
                "weights": {"w_acc": w_acc, "w_sinc": w_sinc},
                "gate": gate,
            },
            result=PreResult(**out),
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": "FILE_NOT_FOUND", "detail": str(e)})
    except Exception as e:
        logger.exception("analyze failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": "INTERNAL_ERROR", "detail": str(e)})

@app.post("/analyze-mint")
async def analyze_and_mint(req: AnalyzeMintReq):
    from analyzer import pre_pipeline
    # 1) 분석 + 게이트
    gate = float(os.getenv("GATE_THRESHOLD", "0.70"))
    res = _call_pre_pipeline_safe(
        text=req.text, denom_mode=req.denom_mode, w_acc=0.5, w_sinc=0.5, gate=gate, pdf_paths=None
    )

    # 2) 게이트 미통과: 점수만 반환
    if not res.get("gate_pass"):
        return {
            "ok": True,
            "minted": False,
            "scores": {"S_acc": res.get("S_acc", 0.0), "S_sinc": res.get("S_sinc", 0.0), "S_pre": res.get("S_pre", 0.0)},
            "pii": {"action": res.get("pii_action", "allow"), "reasons": res.get("pii_reasons", [])},
        }

    # 3) 체인에 남길 메타데이터(마스킹된 본문 + 전체 해시 + 점수)
    masked = res.get("clean_text") or ""
    MAX_LEN = int(os.getenv("TOKENURI_TEXT_MAX", "1000"))
    meta = {
        "name": "Empathy Post",
        "description": "Masked text + scores recorded on-chain",
        "text": masked[:MAX_LEN],
        "text_hash": f"sha256:{sha256(masked.encode('utf-8')).hexdigest()}",
        "scores": {
            "S_acc": round(res.get("S_acc", 0.0), 3),
            "S_sinc": round(res.get("S_sinc", 0.0), 3),
            "S_pre": round(res.get("S_pre", 0.0), 3),
        },
        "version": "v1",
    }

    # 4) 민팅(서명/전송) → tokenId
    to_addr = req.to_address or os.getenv("PUBLIC_ADDRESS")
    try:
        tx_hash = send_mint(to_addr, meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"mint send failed: {e}")
    token_id, _ = wait_token_id(tx_hash)

    return {
        "ok": True,
        "minted": True,
        "tx_hash": tx_hash,
        "explorer": f"https://sepolia.etherscan.io/tx/{tx_hash}",
        "token_id": token_id,
        "scores": meta["scores"],
    }

@app.post("/posts")
async def create_post(p: PostIn):
    # 게이트 미통과 저장 금지
    if not p.scores.gate_pass:
        raise HTTPException(status_code=400, detail="GATE_NOT_PASSED")

    if not USE_DB:
        obj = {
            "title": p.title.strip(),
            "content": p.content.strip(),
            "scores": p.scores.model_dump(),
            "weights": p.weights,
            "files": p.files,
            "meta": p.meta or {},
            "denom_mode": p.denom_mode,
            "gate": p.gate,
            "analysis_id": p.analysis_id or "",
        }
        post_id = _jsonl_append(obj)
        return {"ok": True, "post_id": post_id}

    # DB 모드
    from sqlalchemy.orm import Session  # type: ignore
    with SessionLocal() as db:  # type: ignore
        obj = Post(  # type: ignore
            title=p.title.strip(),
            content=p.content.strip(),
            scores_json=_to_json_str(p.scores),
            weights_json=_to_json_str(p.weights),
            files_json=_to_json_str(p.files),
            meta_json=_to_json_str(p.meta or {}),
            denom_mode=p.denom_mode,
            gate=p.gate,
            analysis_id=p.analysis_id or "",
        )
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return {"ok": True, "post_id": obj.id}

@app.get("/posts/{post_id}", response_model=PostOut)
async def get_post(post_id: int):
    if not USE_DB:
        obj = _jsonl_get(post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")
        return PostOut(
            id=int(obj["id"]),
            title=obj["title"],
            content=obj["content"],
            scores=obj["scores"],
            weights=obj["weights"],
            files=obj["files"],
            meta=obj["meta"],
            denom_mode=obj["denom_mode"],
            gate=obj["gate"],
            analysis_id=obj.get("analysis_id", ""),
            created_at=obj.get("created_at", datetime.utcnow().isoformat() + "Z"),
        )

    # DB 모드
    from sqlalchemy.orm import Session  # type: ignore
    with SessionLocal() as db:  # type: ignore
        obj = db.get(Post, post_id)  # type: ignore
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")
        return PostOut(
            id=obj.id,
            title=obj.title,
            content=obj.content,
            scores=_from_json_str(obj.scores_json, {}),
            weights=_from_json_str(obj.weights_json, {}),
            files=_from_json_str(obj.files_json, []),
            meta=_from_json_str(obj.meta_json, {}),
            denom_mode=obj.denom_mode,
            gate=obj.gate,
            analysis_id=obj.analysis_id or "",
            created_at=(obj.created_at.isoformat() + "Z"),
        )

@app.get("/posts")
async def list_posts(limit: int = 20, offset: int = 0):
    if not USE_DB:
        items_raw = _jsonl_list(limit=limit, offset=offset)
        items = []
        for obj in items_raw:
            sc = obj.get("scores", {})
            items.append(
                {
                    "id": int(obj["id"]),
                    "title": obj["title"],
                    "created_at": obj.get("created_at"),
                    "S_pre": sc.get("S_pre"),
                    "S_sinc": sc.get("S_sinc"),
                    "S_acc": sc.get("S_acc") or sc.get("S_fact"),
                    "gate": obj.get("gate"),
                    "gate_pass": sc.get("gate_pass"),
                }
            )
        return {"ok": True, "items": items, "count": len(items)}

    # DB 모드
    from sqlalchemy.orm import Session  # type: ignore
    with SessionLocal() as db:  # type: ignore
        q = db.query(Post).order_by(Post.id.desc()).offset(offset).limit(limit)  # type: ignore
        items = []
        for obj in q.all():
            scores = _from_json_str(obj.scores_json, {})
            items.append(
                {
                    "id": obj.id,
                    "title": obj.title,
                    "created_at": obj.created_at.isoformat() + "Z",
                    "S_pre": scores.get("S_pre"),
                    "S_sinc": scores.get("S_sinc"),
                    "S_acc": scores.get("S_acc") or scores.get("S_fact"),
                    "gate": obj.gate,
                    "gate_pass": scores.get("gate_pass"),
                }
            )
        return {"ok": True, "items": items, "count": len(items)}
