# main.py
# -*- coding: utf-8 -*-

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

# DB (SQLAlchemy - SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# ────────────────────────────────────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("emotrust-backend")

APP_VERSION = "1.3.0-min"
DB_PATH = "emotrust.db"  # 같은 폴더에 파일 생성

# ────────────────────────────────────────────────────────────────────────────────
# 파이프라인 연결 (사용자 제공 analyzer.py)
# ────────────────────────────────────────────────────────────────────────────────
from analyzer import pre_pipeline

# ────────────────────────────────────────────────────────────────────────────────
# DB 초기화
# ────────────────────────────────────────────────────────────────────────────────
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)

    # 점수/파라미터/메타는 JSON 문자열로 저장(버전 유연성)
    scores_json = Column(Text, nullable=False)      # {S_pre, S_sinc, S_fact, coverage, ...}
    weights_json = Column(Text, nullable=False)     # {w_acc, w_sinc}
    files_json = Column(Text, nullable=False)       # [{name,size}...] or []
    meta_json = Column(Text, nullable=False)        # 프론트·분석 메타

    denom_mode = Column(String(20), default="all")
    gate = Column(Float, default=0.70)
    analysis_id = Column(String(64), index=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
logger.info(f"🗄️ SQLite ready at {Path(DB_PATH).resolve()}")

# ────────────────────────────────────────────────────────────────────────────────
# FastAPI + CORS (개발 편의로 전체 허용)
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
# 스키마
# ────────────────────────────────────────────────────────────────────────────────
class PreResult(BaseModel):
    pii_action: str
    pii_reasons: List[str]
    S_acc: float = Field(ge=0.0, le=1.0)     # 하위호환 이름(=S_fact 또는 0.0)
    S_sinc: float = Field(ge=0.0, le=1.0)
    S_pre: float = Field(ge=0.0, le=1.0)
    S_pre_ext: float = Field(ge=0.0, le=1.0)
    gate_pass: bool
    tokens: int
    matched: int
    total: int
    coverage: float = Field(ge=0.0, le=1.0)
    clean_text: str
    masked: bool
    # B안(증빙) 결과
    S_fact: Optional[float] = Field(default=None)
    need_evidence: bool
    claims: List[str]
    evidence: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    ok: bool
    meta: Dict[str, Any]
    result: PreResult

class ScoresIn(BaseModel):
    S_pre: float
    S_sinc: float
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

# ────────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────────

def await_read_uploadfile(f: UploadFile) -> bytes:
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
        data = await_read_uploadfile(f)
        with open(dst, "wb") as out:
            out.write(data)
        saved_paths.append(str(dst))
    return saved_paths

def to_json_str(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), ensure_ascii=False)
        if hasattr(obj, "dict"):
            return json.dumps(obj.dict(), ensure_ascii=False)
    except Exception:
        pass
    return json.dumps(obj, ensure_ascii=False)

def from_json_str(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

# ────────────────────────────────────────────────────────────────────────────────
# 라우트
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Hello emotrust"

@app.get("/health")
async def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "time": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),      # "all" or "matched"
    w_acc: float = Form(0.5),            # S_fact 가중치
    w_sinc: float = Form(0.5),           # S_sinc 가중치
    gate: float = Form(0.70),
    pdfs: Optional[List[UploadFile]] = File(None),  # 다중 PDF 업로드 지원
):
    try:
        text = f"{title}\n\n{content}".strip() if title else content
        pdf_paths = _save_pdfs(pdfs) if pdfs else []

        out = pre_pipeline(
            text=text,
            denom_mode=denom_mode,
            w_acc=w_acc,
            w_sinc=w_sinc,
            gate=gate,
            pdf_paths=pdf_paths,
        )

        return AnalyzeResponse(
            ok=True,
            meta={
                "title": title,
                "chars": len(text),
                "pdf_count": len(pdf_paths),
                "pdf_paths": pdf_paths,       # 저장용 참고(선택)
                "denom_mode": denom_mode,
                "weights": {"w_acc": w_acc, "w_sinc": w_sinc},
                "gate": gate,
            },
            result=PreResult(**out),
        )

    except FileNotFoundError as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "FILE_NOT_FOUND", "detail": str(e)},
        )
    except Exception as e:
        logger.exception("analyze failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "INTERNAL_ERROR", "detail": str(e)},
        )

@app.post("/posts")
async def create_post(p: PostIn):
    # 기본 방어: gate_pass 안 되면 저장 막기
    if not p.scores.gate_pass:
        raise HTTPException(status_code=400, detail="GATE_NOT_PASSED")

    with SessionLocal() as db:
        obj = Post(
            title=p.title.strip(),
            content=p.content.strip(),
            scores_json=to_json_str(p.scores),
            weights_json=to_json_str(p.weights),
            files_json=to_json_str(p.files),
            meta_json=to_json_str(p.meta or {}),
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
    with SessionLocal() as db:
        obj = db.get(Post, post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")
        return PostOut(
            id=obj.id,
            title=obj.title,
            content=obj.content,
            scores=from_json_str(obj.scores_json, {}),
            weights=from_json_str(obj.weights_json, {}),
            files=from_json_str(obj.files_json, []),
            meta=from_json_str(obj.meta_json, {}),
            denom_mode=obj.denom_mode,
            gate=obj.gate,
            analysis_id=obj.analysis_id or "",
            created_at=(obj.created_at.isoformat() + "Z"),
        )

@app.get("/posts")
async def list_posts(limit: int = 20, offset: int = 0):
    with SessionLocal() as db:
        q = db.query(Post).order_by(Post.id.desc()).offset(offset).limit(limit)
        items = []
        for obj in q.all():
            scores = from_json_str(obj.scores_json, {})
            items.append(
                {
                    "id": obj.id,
                    "title": obj.title,
                    "created_at": obj.created_at.isoformat() + "Z",
                    "S_pre": scores.get("S_pre"),
                    "S_sinc": scores.get("S_sinc"),
                    "S_fact": scores.get("S_fact"),
                    "gate": obj.gate,
                    "gate_pass": scores.get("gate_pass"),
                }
            )
        return {"ok": True, "items": items, "count": len(items)}
