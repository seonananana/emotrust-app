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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select

# 시뮬레이션 체인 유틸 (실체 체인 없음)
from simulate_chain import sim_mint, sim_balance_of

# ────────────────────────────────────────────────────────────────────────────────
# ENV
# ────────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")  # hardhat/.env 로드 제거

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("emotrust-backend")

APP_VERSION = "1.7.0-sim"
DB_PATH = os.getenv("DB_PATH", "emotrust.db")
USE_DB = os.getenv("USE_DB", "true").lower() == "true"   # false → JSONL 저장
AUTO_MINT = os.getenv("AUTO_MINT", "true").lower() == "true"
TOKENURI_TEXT_MAX = int(os.getenv("TOKENURI_TEXT_MAX", "1000"))
# S_THRESHOLD 우선, 없으면 GATE_THRESHOLD 백필
S_THRESHOLD = float(os.getenv("S_THRESHOLD", os.getenv("GATE_THRESHOLD", "0.70")))

# ────────────────────────────────────────────────────────────────────────────────
# 파일(JSONL) 저장 유틸 (USE_DB=false)
# ────────────────────────────────────────────────────────────────────────────────
POSTS_LOG_PATH = os.getenv("POSTS_LOG_PATH", "./data/posts.jsonl")

def _jsonl_append(obj: Dict[str, Any]) -> int:
    os.makedirs(os.path.dirname(POSTS_LOG_PATH), exist_ok=True)
    if "id" not in obj:
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

def _jsonl_update_post(post_id: int, patch: Dict[str, Any]) -> None:
    rows = _jsonl_read_all()
    updated = False
    for i, row in enumerate(rows):
        if int(row.get("id", -1)) == int(post_id):
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(row.get(k), dict):
                    row[k] = {**row[k], **v}
                else:
                    row[k] = v
            rows[i] = row
            updated = True
            break
    if not updated:
        return
    os.makedirs(os.path.dirname(POSTS_LOG_PATH), exist_ok=True)
    tmp = POSTS_LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, POSTS_LOG_PATH)

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

        scores_json = Column(Text, nullable=False)      # {S_pre, S_sinc, S_acc, ...}
        weights_json = Column(Text, nullable=False)     # {w_acc, w_sinc}
        files_json = Column(Text, nullable=False)       # [{name,size}...] or []
        meta_json = Column(Text, nullable=False)        # 프론트/분석 메타

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
# 스키마
# ────────────────────────────────────────────────────────────────────────────────
class PreResult(BaseModel):
    pii_action: str = "none"
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
    # 확장
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
    S_acc: Optional[float] = None
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

class CommentIn(BaseModel):
    author: Optional[str] = "anon"
    text: str = Field(..., description="댓글 내용")

class LikeIn(BaseModel):
    to_address: Optional[str] = Field(None, description="사용자 지갑 주소")

class LikeOut(BaseModel):
    liked: bool
    token_id: Optional[int] = None
    tx_hash: Optional[str] = None
    likes: Optional[int] = None
    
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

def _build_token_meta_from_post(
    title: str,
    content: str,
    scores: Dict[str, Any],
    masked_text: Optional[str] = None
) -> Dict[str, Any]:
    text_for_chain = (masked_text or content or "")[:TOKENURI_TEXT_MAX]
    return {
        "name": "Empathy Post",
        "description": "Masked text + scores recorded (simulated)",
        "text": text_for_chain,
        "text_hash": f"sha256:{sha256((content or '').encode('utf-8')).hexdigest()}",
        "scores": {
            "S_acc": round(float(scores.get("S_acc") or scores.get("S_fact") or 0.0), 3),
            "S_sinc": round(float(scores.get("S_sinc") or 0.0), 3),
            "S_pre": round(float(scores.get("S_pre") or 0.0), 3),
        },
        "version": "v1",
        "mode": "simulated",
    }

def _score_extras_with_comments(scores: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    comments = (meta or {}).get("comments") or []
    count = len(comments)
    bonus = min(0.02 * max(0, count), 0.10)
    try:
        s_pre = float(scores.get("S_pre") or 0.0)
    except Exception:
        s_pre = 0.0
    s_effective = max(0.0, min(1.0, s_pre + bonus))
    return {
        "comment_count": count,
        "comment_bonus": round(bonus, 3),
        "S_effective": round(s_effective, 3),
    }

def _call_pre_pipeline_safe(
    text: str,
    denom_mode: str,
    w_acc: float,
    w_sinc: float,
    gate: float,
) -> Dict[str, Any]:
    # analyzer.pre_pipeline 을 다양한 시그니처로 시도
    from analyzer import pre_pipeline as _pre  # lazy import
    return _pre(text=text, denom_mode=denom_mode, w_acc=w_acc, w_sinc=w_sinc, gate=gate)

# ────────────────────────────────────────────────────────────────────────────────
# 라우트
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Hello emotrust (simulation-only)"

@app.get("/health")
async def health():
    return {"ok": True, "version": APP_VERSION, "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: float = Form(0.70),
):
    pass
except FileNotFoundError as e:
    return JSONResponse(status_code=500, content={"ok": False, "error": "FILE_NOT_FOUND", "detail": str(e)})
except Exception as e:
    logger.exception("analyze failed")
    return JSONResponse(status_code=500, content={"ok": False, "error": "INTERNAL_ERROR", "detail": str(e)})

@app.post("/analyze-and-mint")
async def analyze_and_mint_form(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: Optional[float] = Form(None),
    to_address: Optional[str] = Form(None),
):
    """멀티파트 업로드 → 분석 → 게이트 통과 시 **시뮬레이션 민팅**만 수행"""

        S_pre = float(out.get("S_pre") or out.get("S_pre_ext") or 0.0)
        S_acc = out.get("S_acc") or out.get("S_fact")
        S_sinc = out.get("S_sinc")
        passed = S_pre >= gate_eff

        resp = {
            "ok": True,
            "threshold": gate_eff,
            "scores": {"S_pre": S_pre, "accuracy": S_acc, "authenticity": S_sinc},
            "gate_pass": passed,
            "minted": False,
            "evidence": out.get("evidence"),
            "mode": "simulated",
            "meta": {
                "title": title, "chars": len(text),
                "denom_mode": denom_mode,
                "weights": {"w_acc": w_acc, "w_sinc": w_sinc},
            },
        }

        if not passed:
            return resp

        # 주소 결정 (web3 미사용)
        addr = to_address or os.getenv("PUBLIC_ADDRESS")
        if not addr:
            return JSONResponse(status_code=400, content={"ok": False, "detail": "to_address 또는 PUBLIC_ADDRESS가 필요합니다."})

        # 시뮬 민팅
        tx_hash, token_id = sim_mint(addr)
        resp.update({"minted": True, "tx_hash": tx_hash, "tokenId": token_id})
        return resp

    except Exception as e:
        logger.exception("analyze-and-mint failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": "INTERNAL_ERROR", "detail": str(e)})

@app.post("/analyze-mint")
async def analyze_and_mint(req: AnalyzeMintReq):
    gate = S_THRESHOLD
    res = _call_pre_pipeline_safe(
        text=req.text, denom_mode=req.denom_mode, w_acc=0.5, w_sinc=0.5,
        gate=gate,
    )

    scores = {
        "S_acc": res.get("S_acc", 0.0),
        "S_sinc": res.get("S_sinc", 0.0),
        "S_pre": res.get("S_pre", 0.0),
        "gate_pass": res.get("gate_pass", False),
    }

    # 토큰 보너스(시뮬 밸런스 사용)
    try:
        if req.to_address:
            per = float(os.getenv("NFT_BONUS_PER_TOKEN", "0.02"))
            cap = float(os.getenv("NFT_BONUS_CAP", "0.10"))
            bal = sim_balance_of(req.to_address)
            bonus = min(cap, per * max(0, bal))
            scores["token_bonus"] = bonus
            scores["S_final"] = max(0.0, min(1.0, scores["S_pre"] + bonus))
        else:
            scores["token_bonus"] = 0.0
            scores["S_final"] = scores["S_pre"]
    except Exception:
        scores["token_bonus"] = 0.0
        scores["S_final"] = scores["S_pre"]

    if not res.get("gate_pass"):
        return {"ok": True, "minted": False, "scores": scores, "detail": "Gate not passed; mint skipped"}

    # 주소 필수
    addr = req.to_address or os.getenv("PUBLIC_ADDRESS")
    if not addr:
        raise HTTPException(status_code=400, detail="to_address 또는 PUBLIC_ADDRESS가 필요합니다.")

    tx_hash, token_id = sim_mint(addr)
    return {"minted": True, "tx_hash": tx_hash, "token_id": token_id, "scores": scores, "mode": "simulated"}

# ────────────────────────────────────────────────────────────────────────────────
# Posts
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/posts")
def list_posts(limit: int = 20, offset: int = 0):
    """
    목록 요약: id, title, created_at, S_pre, S_sinc, S_acc, gate, gate_pass, S_effective, likes
    """
    if not USE_DB:
        items_raw = _jsonl_list(limit=limit, offset=offset)
        items = []
        for obj in items_raw:
            sc = obj.get("scores", {}) or {}
            meta = obj.get("meta", {}) or {}
            extras = _score_extras_with_comments(sc, meta)
            items.append({
                "id": int(obj["id"]),
                "title": obj["title"],
                "created_at": obj.get("created_at"),
                "S_pre": sc.get("S_pre"),
                "S_sinc": sc.get("S_sinc"),
                "S_acc": sc.get("S_acc") or sc.get("S_fact"),
                "gate": obj.get("gate"),
                "gate_pass": sc.get("gate_pass"),
                "S_effective": extras["S_effective"],
                "likes": (meta or {}).get("likes"),
            })
        return {"ok": True, "items": items, "count": len(items)}
    else:
        from sqlalchemy.orm import Session  # type: ignore
        with SessionLocal() as db:  # type: ignore
            q = db.query(Post).order_by(Post.created_at.desc()).offset(offset).limit(limit)  # type: ignore
            items: List[Dict[str, Any]] = []
            for obj in q.all():  # type: ignore
                scores = _from_json_str(obj.scores_json, {})  # type: ignore
                meta = _from_json_str(obj.meta_json, {})      # type: ignore
                extras = _score_extras_with_comments(scores, meta)
                items.append({
                    "id": obj.id,
                    "title": obj.title,
                    "created_at": obj.created_at.isoformat() + "Z",
                    "S_pre": scores.get("S_pre"),
                    "S_sinc": scores.get("S_sinc"),
                    "S_acc": scores.get("S_acc") or scores.get("S_fact"),
                    "gate": obj.gate,
                    "gate_pass": scores.get("gate_pass"),
                    "S_effective": extras.get("S_effective"),
                    "likes": meta.get("likes") if meta else None,
                })
            return {"ok": True, "items": items, "count": len(items)}

@app.post("/posts")
async def create_post(p: PostIn):
    if not p.scores.gate_pass:
        raise HTTPException(status_code=400, detail="GATE_NOT_PASSED")

    if not USE_DB:
        obj = {
            "title": p.title.strip(),
            "content": p.content.strip(),
            "scores": p.scores.model_dump() if hasattr(p.scores, "model_dump") else p.scores,
            "weights": p.weights,
            "files": p.files,
            "meta": p.meta or {},
            "denom_mode": p.denom_mode,
            "gate": p.gate,
            "analysis_id": p.analysis_id or "",
        }
        post_id = _jsonl_append(obj)
        saved_title = obj["title"]
        saved_content = obj["content"]
        scores = obj["scores"]
        meta_cur = obj["meta"]
    else:
        from sqlalchemy.orm import Session  # type: ignore
        with SessionLocal() as db:  # type: ignore
            o = Post(  # type: ignore
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
            db.add(o); db.commit(); db.refresh(o)
            post_id = o.id
            saved_title = o.title          # type: ignore
            saved_content = o.content      # type: ignore
            scores = _from_json_str(o.scores_json, {})   # type: ignore
            meta_cur = _from_json_str(o.meta_json, {})   # type: ignore

    # 자동 시뮬 민팅 (실패해도 글은 저장됨)
    minted = False; token_id = None; tx_hash = None; mint_error = None
    if AUTO_MINT:
        try:
            masked_text = None
            if isinstance(meta_cur, dict):
                masked_text = meta_cur.get("masked_text") or meta_cur.get("clean_text")

            meta_token = _build_token_meta_from_post(
                saved_title, saved_content,
                {"S_acc": scores.get("S_acc") or scores.get("S_fact"),
                 "S_sinc": scores.get("S_sinc"),
                 "S_pre": scores.get("S_pre")},
                masked_text=masked_text,
            )

            to_addr = os.getenv("PUBLIC_ADDRESS")
            if not to_addr:
                raise RuntimeError("PUBLIC_ADDRESS not set")

            tx_hash, token_id = sim_mint(to_addr)
            minted = True

            # 저장된 메타에 민팅 결과 기록
            if not USE_DB:
                _jsonl_update_post(int(post_id), {
                    "meta": {**(meta_cur or {}), "minted": True,
                             "mint": {"token_id": token_id, "tx_hash": tx_hash, "mode": "simulated"}}
                })
            else:
                from sqlalchemy.orm import Session  # type: ignore
                with SessionLocal() as db:  # type: ignore
                    o = db.get(Post, int(post_id))  # type: ignore
                    if o:
                        m = _from_json_str(o.meta_json, {})
                        m["minted"] = True
                        m["mint"] = {"token_id": token_id, "tx_hash": tx_hash, "mode": "simulated"}
                        o.meta_json = _to_json_str(m)
                        db.commit()
        except Exception as e:
            mint_error = str(e)

    return {"ok": True, "post_id": int(post_id), "minted": minted,
            "token_id": token_id, "tx_hash": tx_hash, "mint_error": mint_error}

@app.get("/posts/{post_id}", response_model=PostOut)
async def get_post(post_id: int):
    if not USE_DB:
        obj = _jsonl_get(post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")
        sc = obj["scores"]; meta = obj.get("meta") or {}
        extras = _score_extras_with_comments(sc, meta)
        meta["score_extras"] = extras
        sc = {**sc, **extras}
        return PostOut(
            id=int(obj["id"]), title=obj["title"], content=obj["content"],
            scores=sc, weights=obj["weights"], files=obj["files"], meta=meta,
            denom_mode=obj["denom_mode"], gate=obj["gate"],
            analysis_id=obj.get("analysis_id", ""),
            created_at=obj.get("created_at", datetime.utcnow().isoformat() + "Z"),
        )
    else:
        from sqlalchemy.orm import Session  # type: ignore
        with SessionLocal() as db:  # type: ignore
            obj = db.get(Post, post_id)  # type: ignore
            if not obj:
                raise HTTPException(status_code=404, detail="NOT_FOUND")
            scores = _from_json_str(obj.scores_json, {})  # type: ignore
            meta = _from_json_str(obj.meta_json, {})      # type: ignore
            extras = _score_extras_with_comments(scores, meta)
            meta["score_extras"] = extras
            scores = {**scores, **extras}
            return PostOut(
                id=obj.id, title=obj.title, content=obj.content, scores=scores,
                weights=_from_json_str(obj.weights_json, {}),
                files=_from_json_str(obj.files_json, {}), meta=meta,
                denom_mode=obj.denom_mode, gate=obj.gate,
                analysis_id=obj.analysis_id or "",
                created_at=(obj.created_at.isoformat() + "Z"),
            )

# ────────────────────────────────────────────────────────────────────────────────
# Comments
# ====================================================================================
@app.get("/posts/{post_id}/comments")
async def list_comments(post_id: int):
    if not USE_DB:
        obj = _jsonl_get(post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")
        meta = obj.get("meta") or {}
        comments = meta.get("comments") or []
        return {"ok": True, "items": comments}
    else:
        from sqlalchemy.orm import Session  # type: ignore
        with SessionLocal() as db:  # type: ignore
            o = db.get(Post, int(post_id))  # type: ignore
            if not o:
                raise HTTPException(status_code=404, detail="NOT_FOUND")
            meta = _from_json_str(o.meta_json, {})  # type: ignore
            comments = meta.get("comments") or []
            return {"ok": True, "items": comments}


@app.post("/posts/{post_id}/comments")
async def add_comment(post_id: int, c: CommentIn):
    new_item = {
        "id": int(datetime.utcnow().timestamp() * 1000),
        "author": (c.author or "anon"),
        "text": c.text.strip(),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    if not USE_DB:
        obj = _jsonl_get(post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")

        meta = (obj.get("meta") or {})
        comments = meta.get("comments") or []
        comments.append(new_item)
        meta["comments"] = comments

        # 댓글 기반 부가 점수 업데이트 (정의되어 있다고 가정)
        scores_cur = obj.get("scores", {})
        meta["score_extras"] = _score_extras_with_comments(scores_cur, meta)

        _jsonl_update_post(int(post_id), {"meta": meta})
        return {"ok": True, "item": new_item, "count": len(comments)}
    else:
        from sqlalchemy.orm import Session  # type: ignore
        with SessionLocal() as db:  # type: ignore
            o = db.get(Post, int(post_id))  # type: ignore
            if not o:
                raise HTTPException(status_code=404, detail="NOT_FOUND")

            meta = _from_json_str(o.meta_json, {})  # type: ignore
            comments = meta.get("comments") or []
            comments.append(new_item)
            meta["comments"] = comments

            scores_cur = _from_json_str(o.scores_json, {})  # type: ignore
            meta["score_extras"] = _score_extras_with_comments(scores_cur, meta)

            o.meta_json = _to_json_str(meta)  # type: ignore
            db.commit()
            return {"ok": True, "item": new_item, "count": len(comments)}


# ============================================================
# Likes (공감) + 시뮬 토큰
# ============================================================
@app.post("/posts/{post_id}/like", response_model=LikeOut)
async def like_post(post_id: int, data: LikeIn):
    def _resolve_addr(given: Optional[str]) -> Optional[str]:
        if given:
            return given
        return os.getenv("PUBLIC_ADDRESS")

    if not USE_DB:
        obj = _jsonl_get(post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")

        meta = obj.get("meta") or {}
        likes = int(meta.get("likes") or 0) + 1
        meta["likes"] = likes

        tx_hash: Optional[str] = None
        token_id: Optional[int] = None

        to_addr = _resolve_addr(data.to_address)
        if to_addr:
            # 공감 시뮬 민트
            tx_hash, token_id = sim_mint(to_addr)
            mints = meta.get("like_mints") or []
            mints.append({
                "addr": to_addr,
                "tx_hash": tx_hash,
                "token_id": token_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
            meta["like_mints"] = mints

        _jsonl_update_post(int(post_id), {"meta": meta})
        return LikeOut(liked=True, token_id=token_id, tx_hash=tx_hash, likes=likes)

    else:
        from sqlalchemy.orm import Session  # type: ignore
        with SessionLocal() as db:  # type: ignore
            o = db.get(Post, int(post_id))  # type: ignore
            if not o:
                raise HTTPException(status_code=404, detail="NOT_FOUND")

            meta = _from_json_str(o.meta_json, {})  # type: ignore
            likes = int(meta.get("likes") or 0) + 1
            meta["likes"] = likes

            tx_hash: Optional[str] = None
            token_id: Optional[int] = None

            to_addr = _resolve_addr(data.to_address)
            if to_addr:
                tx_hash, token_id = sim_mint(to_addr)
                mints = meta.get("like_mints") or []
                mints.append({
                    "addr": to_addr,
                    "tx_hash": tx_hash,
                    "token_id": token_id,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                })
                meta["like_mints"] = mints

            o.meta_json = _to_json_str(meta)  # type: ignore
            db.commit()
            return LikeOut(liked=True, token_id=token_id, tx_hash=tx_hash, likes=likes)
