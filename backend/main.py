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

from dotenv import load_dotenv
from simulate_chain import sim_mint, sim_balance_of

# ────────────────────────────────────────────────────────────────────────────────
# ENV 로드 (backend/.env → hardhat/.env 순서로)
# ────────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent  # backend/
load_dotenv(BASE / ".env")
load_dotenv(BASE.parent / "hardhat" / ".env", override=False)

# ────────────────────────────────────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("emotrust-backend")

APP_VERSION = "1.4.1"
DB_PATH = os.getenv("DB_PATH", "emotrust.db")
USE_DB = os.getenv("USE_DB", "true").lower() == "true"   # false면 파일(JSONL) 저장으로 대체

# --- Auto-mint settings ---
AUTO_MINT = os.getenv("AUTO_MINT", "true").lower() == "true"  # 기본: 자동 민팅 ON
TOKENURI_TEXT_MAX = int(os.getenv("TOKENURI_TEXT_MAX", "1000"))

def _build_token_meta_from_post(
    title: str,
    content: str,
    scores: Dict[str, Any],
    masked_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    NFT 메타데이터 생성: 마스킹 텍스트가 있으면 사용, 없으면 본문 일부/해시만 기록.
    """
    text_for_chain = (masked_text or content or "")[:TOKENURI_TEXT_MAX]
    return {
        "name": "Empathy Post",
        "description": "Masked text + scores recorded on-chain",
        "text": text_for_chain,
        "text_hash": f"sha256:{sha256((content or '').encode('utf-8')).hexdigest()}",
        "scores": {
            "S_acc": round(float(scores.get("S_acc") or scores.get("S_fact") or 0.0), 3),
            "S_sinc": round(float(scores.get("S_sinc") or 0.0), 3),
            "S_pre": round(float(scores.get("S_pre") or 0.0), 3),
        },
        "version": "v1",
    }

# ────────────────────────────────────────────────────────────────────────────────
# 파일(JSONL) 저장 유틸 (USE_DB=false일 때 사용)
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
    """
    posts.jsonl 전체를 읽어 해당 id를 찾아 병합 업데이트 후 파일을 덮어쓴다.
    patch는 dict로 들어오며, 중첩 dict(meta 등)는 얕은 병합.
    """
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
# 스키마
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
    # 확장 필드
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

# 업로드 PDF를 (파일명, 바이트) 튜플 리스트로 수집
def _collect_pdf_blobs(pdfs: Optional[List[UploadFile]]) -> List[Tuple[str, bytes]]:
    pdf_blobs: List[Tuple[str, bytes]] = []
    for f in (pdfs or []):
        try:
            blob = f.file.read()
            print(f"✅ PDF: {f.filename}, Size: {len(blob)} bytes")
            pdf_blobs.append((f.filename, blob))
        except Exception as e:
            print(f"❌ Error reading PDF {f.filename}: {e}")
    return pdf_blobs

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
    text: str,
    denom_mode: str,
    w_acc: float,
    w_sinc: float,
    gate: float,
    pdf_paths: Optional[List[str]],
    pdf_blobs: Optional[List[Tuple[str, bytes]]] = None,
) -> Dict[str, Any]:
    """
    pre_pipeline 시그니처가 버전에 따라
      - pdf_blobs / pdf_paths 둘 다 받거나
      - 하나만 받거나
      - 전혀 안 받을 수도 있어서
    가장 풍부한 시도 → 단순 시도 순으로 호출한다.
    """
    from analyzer import pre_pipeline as _pre  # lazy import

    # 1) (text, denom_mode, w_acc, w_sinc, gate, pdf_paths, pdf_blobs)
    try:
        return _pre(
            text=text, denom_mode=denom_mode,
            w_acc=w_acc, w_sinc=w_sinc, gate=gate,
            pdf_paths=pdf_paths, pdf_blobs=pdf_blobs
        )
    except TypeError:
        pass

    # 2) (text, denom_mode, w_acc, w_sinc, gate, pdf_blobs)
    try:
        return _pre(
            text=text, denom_mode=denom_mode,
            w_acc=w_acc, w_sinc=w_sinc, gate=gate,
            pdf_blobs=pdf_blobs
        )
    except TypeError:
        pass

    # 3) (text, denom_mode, w_acc, w_sinc, gate, pdf_paths)
    try:
        return _pre(
            text=text, denom_mode=denom_mode,
            w_acc=w_acc, w_sinc=w_sinc, gate=gate,
            pdf_paths=pdf_paths
        )
    except TypeError:
        pass

    # 4) (text, denom_mode, w_acc, w_sinc, gate)
    return _pre(text=text, denom_mode=denom_mode, w_acc=w_acc, w_sinc=w_sinc, gate=gate)

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
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: float = Form(0.70),
    pdfs: Optional[List[UploadFile]] = File(None),
):
    try:
        text = f"{title}\n\n{content}".strip() if title else content

        # PDF 경로/바이트 수집
        pdf_paths = _save_pdfs(pdfs) if pdfs else []
        pdf_blobs = _collect_pdf_blobs(pdfs) if pdfs else []

        print("🔥 Calling score_with_pdf with blobs:", pdf_blobs)  # 디버깅 로그

        out = _call_pre_pipeline_safe(
            text=text, denom_mode=denom_mode, w_acc=w_acc, w_sinc=w_sinc, gate=gate,
            pdf_paths=pdf_paths, pdf_blobs=pdf_blobs
        )

        return AnalyzeResponse(
            ok=True,
            meta={
                "title": title,
                "chars": len(text),
                "pdf_count": len(pdf_blobs) if pdf_blobs else len(pdf_paths),
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
        # 👇 기존 analyze 아래에 추가 (main.py)
S_THRESHOLD = float(os.getenv("S_THRESHOLD", "0.70"))

@app.post("/analyze-and-mint")
async def analyze_and_mint_form(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: Optional[float] = Form(None),  # 개별 요청에서 오버라이드 가능
    pdfs: Optional[List[UploadFile]] = File(None),
    to_address: Optional[str] = Form(None),  # 지정 안 하면 아래에서 자동 추론
):
    """
    멀티파트 업로드 전용 엔드포인트:
    - title/content + pdfs[] 업로드
    - 분석 + 게이트 판단
    - 통과 시 시뮬 민팅(simulate_chain)
    """
    try:
        # 1) 텍스트/파일 수집
        text = f"{title}\n\n{content}".strip() if title else content
        pdf_blobs = _collect_pdf_blobs(pdfs)  # (filename, bytes) 리스트

        # 2) 분석 파이프라인 호출 (기존 analyze와 동일 방식)
        gate_eff = float(gate if gate is not None else S_THRESHOLD)
        out = _call_pre_pipeline_safe(
            text=text,
            denom_mode=denom_mode,
            w_acc=w_acc,
            w_sinc=w_sinc,
            gate=gate_eff,
            pdf_paths=[],          # 경로 저장 안 씀
            pdf_blobs=pdf_blobs,   # 핵심
        )

        # 3) 점수/게이트
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
            "meta": {
                "title": title,
                "chars": len(text),
                "pdf_count": len(pdf_blobs),
                "denom_mode": denom_mode,
                "weights": {"w_acc": w_acc, "w_sinc": w_sinc},
            },
        }

        if not passed:
            return resp  # 게이트 미통과 → 민팅 스킵

        # 4) 민팅 대상 주소 결정 (PUBLIC_ADDRESS → PRIVATE_KEY 유도 → 폼 입력)
        addr = to_address or os.getenv("PUBLIC_ADDRESS")
        if not addr:
            pk = os.getenv("PRIVATE_KEY")
            if pk:
                try:
                    from web3 import Web3
                    addr = Web3().eth.account.from_key(pk).address
                except Exception:
                    addr = None

        # 5) 시뮬/실체인 분기 (기본은 시뮬)
        simulate = os.getenv("EMOTRUST_SIMULATE_CHAIN", "1") == "1"
        if simulate:
            if not addr:
                # 시뮬이라도 수령 주소가 없으면 더미로 진행 가능하게 처리(선호: 주소 요구)
                # 여기서는 명시적으로 주소 필요로 할게
                return JSONResponse(status_code=400, content={"ok": False, "detail": "to_address가 필요합니다."})
            tx_hash, token_id = sim_mint(addr)
            resp.update({"minted": True, "tx_hash": tx_hash, "tokenId": token_id, "mode": "simulated"})
            return resp

        # 실체인 (운영 전환 시)
        if not addr:
            return JSONResponse(status_code=400, content={"ok": False, "detail": "to_address가 필요합니다."})
        from mint.mint import send_mint, wait_token_id  # lazy import
        tx_hash = send_mint(addr, _build_token_meta_from_post(title, content, {"S_acc": S_acc, "S_sinc": S_sinc, "S_pre": S_pre}))
        token_id, _ = wait_token_id(tx_hash)
        resp.update({"minted": True, "tx_hash": tx_hash, "tokenId": token_id, "mode": "onchain"})
        return resp

    except Exception as e:
        logger.exception("analyze-and-mint failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": "INTERNAL_ERROR", "detail": str(e)})

@app.post("/analyze-mint")
async def analyze_and_mint(req: AnalyzeMintReq):
    gate = float(os.getenv("GATE_THRESHOLD", "0.70"))
    res = _call_pre_pipeline_safe(
        text=req.text,
        denom_mode=req.denom_mode,
        w_acc=0.5,
        w_sinc=0.5,
        gate=gate,
        pdf_paths=None,
    )

    scores = {
        "S_acc": res.get("S_acc", 0.0),
        "S_sinc": res.get("S_sinc", 0.0),
        "S_pre": res.get("S_pre", 0.0),
        "gate_pass": res.get("gate_pass", False),
    }

    # 토큰 보너스 적용(시뮬 모드도 동일)
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
        return {
            "ok": True,
            "minted": False,
            "scores": scores,
            "detail": "Gate not passed; mint skipped",
        }

    # --- 시뮬 모드 분기 ---
    if os.getenv("EMOTRUST_SIMULATE_CHAIN", "0") == "1":
        if not req.to_address:
            return {"minted": False, "detail": "user_address(to_address)가 필요합니다."}
        tx_hash, token_id = sim_mint(req.to_address)
        return {
            "minted": True,
            "tx_hash": tx_hash,
            "token_id": token_id,
            "scores": scores,
        }

    # --- 실제 민팅 (EMOTRUST_DISABLE_MINT=0 && EMOTRUST_SIMULATE_CHAIN=0) ---
    from mint.mint import send_mint, wait_token_id  # lazy import (운영 전환 시)
    if not req.to_address:
        raise HTTPException(status_code=400, detail="to_address가 필요합니다.")
    m1 = send_mint(req.to_address)
    m2 = wait_token_id(m1.tx_hash)
    return {
        "minted": True,
        "tx_hash": m1.tx_hash,
        "token_id": m2.token_id,
        "scores": scores,
    }

@app.post("/posts")
async def create_post(p: PostIn):
    # 게이트 미통과는 저장 금지(기존 정책 유지)
    if not p.scores.gate_pass:
        raise HTTPException(status_code=400, detail="GATE_NOT_PASSED")

    # 1) 글 저장 (파일 모드/DB 모드 공통)
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
        # 파일 모드에선 바로 쓰던 데이터로 진행
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
            db.add(o)
            db.commit()
            db.refresh(o)
            post_id = o.id
            saved_title = o.title          # type: ignore
            saved_content = o.content      # type: ignore
            scores = _from_json_str(o.scores_json, {})   # type: ignore
            meta_cur = _from_json_str(o.meta_json, {})   # type: ignore

    # 2) 자동 민팅 시도 (성공/실패와 무관하게 글은 이미 저장됨)
    minted = False
    token_id = None
    tx_hash = None
    explorer = None
    mint_error = None

    if AUTO_MINT:
        try:
            # analyzer 결과의 마스킹 텍스트가 meta에 들어왔다면 사용
            masked_text = None
            if isinstance(meta_cur, dict):
                masked_text = meta_cur.get("masked_text") or meta_cur.get("clean_text")

            # 토큰 메타 구성
            meta_token = _build_token_meta_from_post(
                saved_title, saved_content,
                {
                    "S_acc": scores.get("S_acc") or scores.get("S_fact"),
                    "S_sinc": scores.get("S_sinc"),
                    "S_pre": scores.get("S_pre"),
                },
                masked_text=masked_text,
            )

            # 수령 주소: PUBLIC_ADDRESS > PRIVATE_KEY 파생
            to_addr = os.getenv("PUBLIC_ADDRESS")
            if not to_addr:
                pk = os.getenv("PRIVATE_KEY")
                if pk:
                    from web3 import Web3
                    to_addr = Web3().eth.account.from_key(pk).address
            if not to_addr:
                raise RuntimeError("PUBLIC_ADDRESS not set")

            # --- B안: 시뮬 민팅 분기 ---
            if os.getenv("EMOTRUST_SIMULATE_CHAIN", "0") == "1":
                tx_hash, token_id = sim_mint(to_addr)
                minted = True
                explorer = None  # 시뮬이므로 익스플로러 링크 없음
            else:
                # 실제 민팅 (운영 전환 시)
                from mint.mint import send_mint, wait_token_id  # lazy import
                tx_hash = send_mint(to_addr, meta_token)
                token_id, _ = wait_token_id(tx_hash)
                minted = True
                explorer = f"https://sepolia.etherscan.io/tx/{tx_hash}"

            # 3) 민팅 결과를 저장 데이터에 반영
            if not USE_DB:
                _jsonl_update_post(int(post_id), {
                    "meta": {
                        **(meta_cur or {}),
                        "minted": True,
                        "mint": {"token_id": token_id, "tx_hash": tx_hash, "explorer": explorer},
                    }
                })
            else:
                from sqlalchemy.orm import Session  # type: ignore
                with SessionLocal() as db:  # type: ignore
                    o = db.get(Post, int(post_id))  # type: ignore
                    if o:
                        m = _from_json_str(o.meta_json, {})
                        m["minted"] = True
                        m["mint"] = {"token_id": token_id, "tx_hash": tx_hash, "explorer": explorer}
                        o.meta_json = _to_json_str(m)
                        db.commit()

        except Exception as e:
            mint_error = str(e)  # 실패해도 글은 저장됐으므로 minted=False로 응답

    return {
        "ok": True,
        "post_id": int(post_id),
        "minted": minted,
        "token_id": token_id,
        "tx_hash": tx_hash,
        "explorer": explorer,
        "mint_error": mint_error,
    }

# main.py
# -*- coding: utf-8 -*-

@app.get("/posts/{post_id}", response_model=PostOut)
async def get_post(post_id: int):
    if not USE_DB:
        obj = _jsonl_get(post_id)
        if not obj:
            raise HTTPException(status_code=404, detail="NOT_FOUND")
        # 댓글 보너스 반영
        sc = obj["scores"]
        meta = obj["meta"]
        meta = meta or {}
        extras = _score_extras_with_comments(sc, meta)
        meta["score_extras"] = extras
        sc = {**sc, **extras}

        return PostOut(
            id=int(obj["id"]),
            title=obj["title"],
            content=obj["content"],
            scores=sc,
            weights=obj["weights"],
            files=obj["files"],
            meta=meta,
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

        scores = _from_json_str(obj.scores_json, {})  # type: ignore
        meta = _from_json_str(obj.meta_json, {})      # type: ignore
        extras = _score_extras_with_comments(scores, meta)
        meta["score_extras"] = extras
        scores = {**scores, **extras}

        return PostOut(
            id=obj.id,
            title=obj.title,
            content=obj.content,
            scores=scores,
            weights=_from_json_str(obj.weights_json, {}),
            files=_from_json_str(obj.files_json, {}),
            meta=meta,
            denom_mode=obj.denom_mode,
            gate=obj.gate,
            analysis_id=obj.analysis_id or "",
            created_at=(obj.created_at.isoformat() + "Z"),
        )

async def list_posts(limit: int = 20, offset: int = 0):
     if not USE_DB:
         items_raw = _jsonl_list(limit=limit, offset=offset)
         items = []
         for obj in items_raw:
-            sc = obj.get("scores", {})
+            meta = obj.get("meta", {}) or {}
+            sc = obj.get("scores", {}) or {}
+            extras = _score_extras_with_comments(sc, meta)
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
+                    "S_effective": extras["S_effective"],
+                    "likes": (meta or {}).get("likes"),
                     # minted 여부/정보는 상세(meta)에서 확인 가능. 필요하다면 여기도 풀어줄 수 있음.
                 }
             )
         return {"ok": True, "items": items, "count": len(items)}
 
@@
         for obj in q.all():
-            scores = _from_json_str(obj.scores_json, {})
+            scores = _from_json_str(obj.scores_json, {})
+            meta = _from_json_str(obj.meta_json, {})
+            extras = _score_extras_with_comments(scores, meta)
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
+                    "S_effective": extras["S_effective"],
+                    "likes": (meta or {}).get("likes"),
                 }
             )
         return {"ok": True, "items": items, "count": len(items)}
+
+# ─────────────────────────────────────────────────────────
+# 댓글 목록
+# ─────────────────────────────────────────────────────────
+@app.get("/posts/{post_id}/comments")
+async def list_comments(post_id: int):
+    if not USE_DB:
+        obj = _jsonl_get(post_id)
+        if not obj:
+            raise HTTPException(status_code=404, detail="NOT_FOUND")
+        meta = obj.get("meta") or {}
+        comments = meta.get("comments") or []
+        return {"ok": True, "items": comments}
+    else:
+        from sqlalchemy.orm import Session  # type: ignore
+        with SessionLocal() as db:  # type: ignore
+            o = db.get(Post, int(post_id))  # type: ignore
+            if not o:
+                raise HTTPException(status_code=404, detail="NOT_FOUND")
+            meta = _from_json_str(o.meta_json, {})  # type: ignore
+            comments = meta.get("comments") or []
+            return {"ok": True, "items": comments}
+
+# ─────────────────────────────────────────────────────────
+# 댓글 등록 (점수 보너스만 반영; 토큰 민팅 없음)
+# ─────────────────────────────────────────────────────────
+@app.post("/posts/{post_id}/comments")
+async def add_comment(post_id: int, c: CommentIn):
+    new_item = {
+        "id": int(datetime.utcnow().timestamp() * 1000),
+        "author": (c.author or "anon"),
+        "text": c.text.strip(),
+        "created_at": datetime.utcnow().isoformat() + "Z",
+    }
+    if not USE_DB:
+        obj = _jsonl_get(post_id)
+        if not obj:
+            raise HTTPException(status_code=404, detail="NOT_FOUND")
+        meta = (obj.get("meta") or {})
+        comments = meta.get("comments") or []
+        comments.append(new_item)
+        meta["comments"] = comments
+        # 댓글 보너스 재계산
+        scores_cur = obj.get("scores", {})
+        meta["score_extras"] = _score_extras_with_comments(scores_cur, meta)
+        _jsonl_update_post(int(post_id), {"meta": meta})
+        return {"ok": True, "item": new_item, "count": len(comments)}
+    else:
+        from sqlalchemy.orm import Session  # type: ignore
+        with SessionLocal() as db:  # type: ignore
+            o = db.get(Post, int(post_id))  # type: ignore
+            if not o:
+                raise HTTPException(status_code=404, detail="NOT_FOUND")
+            meta = _from_json_str(o.meta_json, {})  # type: ignore
+            comments = meta.get("comments") or []
+            comments.append(new_item)
+            meta["comments"] = comments
+            # 댓글 보너스 재계산
+            scores_cur = _from_json_str(o.scores_json, {})  # type: ignore
+            meta["score_extras"] = _score_extras_with_comments(scores_cur, meta)
+            o.meta_json = _to_json_str(meta)  # type: ignore
+            db.commit()
+            return {"ok": True, "item": new_item, "count": len(comments)}
+
+# ─────────────────────────────────────────────────────────
+# 좋아요(+선택적 공감 토큰 민팅; 점수 영향 없음)
+# ─────────────────────────────────────────────────────────
+@app.post("/posts/{post_id}/like", response_model=LikeOut)
+async def like_post(post_id: int, data: LikeIn = LikeIn()):
+    simulate = os.getenv("EMOTRUST_SIMULATE_CHAIN", "1") == "1"
+    like_mint_on = os.getenv("EMOTRUST_LIKE_MINT", "1") == "1"
+
+    def _resolve_addr(given: Optional[str]) -> Optional[str]:
+        if given:
+            return given
+        addr = os.getenv("PUBLIC_ADDRESS")
+        if addr:
+            return addr
+        pk = os.getenv("PRIVATE_KEY")
+        if pk:
+            try:
+                from web3 import Web3
+                return Web3().eth.account.from_key(pk).address
+            except Exception:
+                return None
+        return None
+
+    if not USE_DB:
+        obj = _jsonl_get(post_id)
+        if not obj:
+            raise HTTPException(status_code=404, detail="NOT_FOUND")
+        meta = obj.get("meta") or {}
+        likes = int(meta.get("likes") or 0) + 1
+        meta["likes"] = likes
+
+        minted = False
+        tx_hash = None
+        token_id = None
+
+        if like_mint_on and simulate:
+            to_addr = _resolve_addr(data.to_address)
+            if to_addr:
+                tx_hash, token_id = sim_mint(to_addr)  # 공감 토큰 시뮬
+                minted = True
+                mints = meta.get("like_mints") or []
+                mints.append({
+                    "addr": to_addr,
+                    "tx_hash": tx_hash,
+                    "token_id": token_id,
+                    "created_at": datetime.utcnow().isoformat() + "Z",
+                })
+                meta["like_mints"] = mints
+
+        _jsonl_update_post(int(post_id), {"meta": meta})
+        return LikeOut(ok=True, likes=likes, minted=minted, tx_hash=tx_hash, token_id=token_id)
+    else:
+        from sqlalchemy.orm import Session  # type: ignore
+        with SessionLocal() as db:  # type: ignore
+            o = db.get(Post, int(post_id))  # type: ignore
+            if not o:
+                raise HTTPException(status_code=404, detail="NOT_FOUND")
+            meta = _from_json_str(o.meta_json, {})  # type: ignore
+            likes = int(meta.get("likes") or 0) + 1
+            meta["likes"] = likes
+
+            minted = False
+            tx_hash = None
+            token_id = None
+
+            if like_mint_on and simulate:
+                to_addr = _resolve_addr(data.to_address)
+                if to_addr:
+                    tx_hash, token_id = sim_mint(to_addr)
+                    minted = True
+                    mints = meta.get("like_mints") or []
+                    mints.append({
+                        "addr": to_addr,
+                        "tx_hash": tx_hash,
+                        "token_id": token_id,
+                        "created_at": datetime.utcnow().isoformat() + "Z",
+                    })
+                    meta["like_mints"] = mints
+
+            o.meta_json = _to_json_str(meta)  # type: ignore
+            db.commit()
+            return LikeOut(ok=True, likes=likes, minted=minted, tx_hash=tx_hash, token_id=token_id)
