# main.py
# -*- coding: utf-8 -*-

import os
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
import requests

# ────────────────────────────────────────────────────────────────────────────────
# 환경 / 로깅
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("emotrust-backend")

# 프런트 CORS
ALLOW_ORIGINS_ENV = os.getenv("ALLOW_ORIGINS") or "*"
ALLOW_ORIGINS = [o.strip() for o in ALLOW_ORIGINS_ENV.split(",") if o.strip()]

# ngrok/도메인
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or os.getenv("NGROK_PUBLIC_URL") or "").strip() or None
NGROK_API_URL = os.getenv("NGROK_API_URL", "http://127.0.0.1:4040").strip()

APP_VERSION = os.getenv("APP_VERSION", "1.2.0")

# ────────────────────────────────────────────────────────────────────────────────
# 파이프라인 연결
# ────────────────────────────────────────────────────────────────────────────────
# analyzer.py: PII → 전처리 → 진정성(사전) → PDF 팩트체크 → 결합
from analyzer import pre_pipeline

# ────────────────────────────────────────────────────────────────────────────────
# FastAPI 초기화 + CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="emotrust-backend", version=APP_VERSION)

if len(ALLOW_ORIGINS) == 1 and ALLOW_ORIGINS[0] == "*":
    # 개발 편의: 전체 허용
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("🌐 CORS: allow_origin_regex='.*' (DEV)")
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"🌐 CORS: allow_origins={ALLOW_ORIGINS}")

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
    S_fact: Optional[float] = Field(default=None)  # None이면 검증 불가
    need_evidence: bool
    claims: List[str]
    evidence: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    ok: bool
    meta: Dict[str, Any]
    result: PreResult

# ────────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────────
def pick_ngrok_public_url(tunnels_json: dict) -> Optional[str]:
    tunnels = tunnels_json.get("tunnels", []) or []
    # HTTPS 우선
    for t in tunnels:
        url = str(t.get("public_url", ""))
        if url.startswith("https://"):
            return url.rstrip("/")
    # HTTP 폴백
    for t in tunnels:
        url = str(t.get("public_url", ""))
        if url.startswith("http://"):
            return url.rstrip("/")
    return None

def _save_pdfs(pdfs: Optional[List[UploadFile]]) -> List[str]:
    """업로드된 PDF들을 임시 폴더에 저장하고 파일 경로 리스트를 반환."""
    if not pdfs:
        return []
    saved_paths: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="emotrust_pdf_")
    for i, f in enumerate(pdfs):
        # 확장자 보정
        name = f.filename or f"evidence_{i}.pdf"
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        dst = Path(tmpdir) / name
        data = await_read_uploadfile(f)
        with open(dst, "wb") as out:
            out.write(data)
        saved_paths.append(str(dst))
    return saved_paths

def await_read_uploadfile(f: UploadFile) -> bytes:
    """
    UploadFile.read()는 sync 컨텍스트에서 직접 호출 시 경고가 뜰 수 있어
    여기서는 파일 객체에서 raw bytes를 안전하게 얻도록 분리.
    """
    try:
        return f.file.read()
    finally:
        try:
            f.file.seek(0)
        except Exception:
            pass

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

@app.get("/ngrok-url")
async def get_ngrok_url():
    # 1) 환경변수로 고정 도메인/URL 지정된 경우
    if PUBLIC_BASE_URL:
        url = PUBLIC_BASE_URL.rstrip("/")
        return {
            "ngrok_url": url,
            "source": "env",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "endpoints": {"root": f"{url}/", "analyze": f"{url}/analyze"},
        }

    # 2) 로컬 ngrok API 조회
    tunnels_api = f"{NGROK_API_URL.rstrip('/')}/api/tunnels"
    try:
        resp = requests.get(tunnels_api, timeout=2.5)
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ngrok 로컬 API에 연결 실패",
                "hint": "ngrok가 실행 중인지 확인하세요. 예) ngrok http 8000",
                "checked_url": tunnels_api,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"error": f"ngrok API 응답 코드 {resp.status_code}", "body": resp.text[:300]},
        )

    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail={"error": "ngrok API가 JSON이 아닙니다.", "body": resp.text[:300]},
        )

    url = pick_ngrok_public_url(data)
    if not url:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "활성화된 ngrok 퍼블릭 URL을 찾지 못했습니다.",
                "hint": "ngrok 터널이 생성되었는지 확인하세요.",
                "tunnels": data.get("tunnels", []),
            },
        )

    return {
        "ngrok_url": url,
        "source": "ngrok",
        "tunnel_count": len(data.get("tunnels", []) or []),
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "endpoints": {"root": f"{url}/", "analyze": f"{url}/analyze"},
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),      # "all" or "matched"
    w_acc: float = Form(0.5),           # S_fact 가중치
    w_sinc: float = Form(0.5),          # S_sinc 가중치
    gate: float = Form(0.70),
    pdfs: Optional[List[UploadFile]] = File(None),  # 다중 PDF 업로드 지원
):
    """
    입력:
      - title, content: 분석 텍스트
      - denom_mode: 'all' | 'matched'
      - w_acc, w_sinc: 가중치
      - gate: 최종 게이트 임계값
      - pdfs: 증빙 PDF 여러 개
    출력:
      - analyzer.pre_pipeline 결과를 그대로 result에 담아 반환
    """
    try:
        text = f"{title}\n\n{content}".strip() if title else content
        pdf_paths = _save_pdfs(pdfs) if pdfs else None

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
                "pdf_count": len(pdf_paths or []),
                "denom_mode": denom_mode,
                "weights": {"w_acc": w_acc, "w_sinc": w_sinc},
                "gate": gate,
            },
            result=PreResult(**out),
        )

    except FileNotFoundError as e:
        # 예: 사전 CSV 경로 문제 등
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
