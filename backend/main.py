# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import requests
import logging
import os
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────
# 환경 설정
# ─────────────────────────────────────────────────────────────────────
load_dotenv()

PORT = int(os.getenv("PORT", "8000"))

# PUBLIC_BASE_URL이 있으면 고정 도메인(ngrok 예약 도메인/실배포)을 사용
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or os.getenv("NGROK_PUBLIC_URL") or "").strip().rstrip("/")

# CORS: 콤마로 여러 개 허용 가능, 비어있으면 개발 편의 위해 *
ALLOW_ORIGINS_RAW = os.getenv("ALLOW_ORIGINS", "*").strip()
ALLOW_ORIGINS: List[str] = ["*"] if ALLOW_ORIGINS_RAW == "*" else [o.strip() for o in ALLOW_ORIGINS_RAW.split(",") if o.strip()]

# ─────────────────────────────────────────────────────────────────────
# 앱/미들웨어
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="emotrust-backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == ["*"] else ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────
# 유틸: ngrok 공개 URL 탐지
# ─────────────────────────────────────────────────────────────────────
def _pick_https_tunnel(tunnels: list, port: int) -> Optional[str]:
    """
    ngrok 로컬 API(/api/tunnels) 응답에서
    1) proto == https 이고
    2) config.addr가 :{port}로 끝나는 터널을 우선 선택.
       없으면 첫 번째 https 터널 반환.
    """
    https_candidates = []
    for t in tunnels or []:
        proto = t.get("proto")
        public_url = t.get("public_url", "")
        cfg = t.get("config", {}) or {}
        addr = str(cfg.get("addr", ""))

        if proto != "https":
            continue
        https_candidates.append(public_url)

        if addr.endswith(f":{port}") or addr.endswith(f"127.0.0.1:{port}") or addr.endswith(f"localhost:{port}"):
            return public_url

    return https_candidates[0] if https_candidates else None


def discover_public_url(port: int = PORT) -> Tuple[Optional[str], str]:
    """
    (url, source) 반환
    source = "env" | "ngrok" | "none"
    """
    # 1) 고정 도메인(환경변수) 우선
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL, "env"

    # 2) ngrok 로컬 API에서 현재 퍼블릭 URL 조회
    NGROK_API = os.getenv("NGROK_API", "http://127.0.0.1:4040/api/tunnels")
    try:
        r = requests.get(NGROK_API, timeout=1.5)
        data = r.json()
        url = _pick_https_tunnel(data.get("tunnels", []), port)
        if url:
            return url.rstrip("/"), "ngrok"
    except Exception as e:
        logging.debug(f"ngrok API 조회 실패: {e}")

    # 3) 없으면 None
    return None, "none"

# ─────────────────────────────────────────────────────────────────────
# 헬스체크/유틸 엔드포인트
# ─────────────────────────────────────────────────────────────────────
@app.get("/hello")
def hello():
    return {
        "status": "ok",
        "server_time": datetime.utcnow().isoformat() + "Z",
        "port": PORT,
    }

@app.get("/ngrok-url")
def get_ngrok_url():
    url, source = discover_public_url(PORT)
    return {
        "ngrok_url": url,          # 고정 도메인 또는 현재 ngrok https 주소 또는 None
        "source": source,          # "env" | "ngrok" | "none"
        "port": PORT,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

# ─────────────────────────────────────────────────────────────────────
# (참고) 기존 /analyze 등 분석 엔드포인트는 그대로 두세요.
# 이 파일 맨 아래에 이미 구현되어 있다면 그대로 유지하면 됩니다.
# ─────────────────────────────────────────────────────────────────────
