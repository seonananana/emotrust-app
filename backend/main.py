# app.py
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
import requests
import logging
import os
import json
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────────
# 환경/설정
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ .env에 OPENROUTER_API_KEY가 없습니다.")

MODEL_NAME = os.getenv("LLM_MODEL", "mistralai/mixtral-8x7b-instruct")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 프런트 접근 오리진(없으면 *)
ALLOW_ORIGINS = (os.getenv("ALLOW_ORIGINS") or "*").split(",")

# ngrok/도메인 오버라이드
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL") or os.getenv("NGROK_PUBLIC_URL")
NGROK_API_URL = os.getenv("NGROK_API_URL", "http://127.0.0.1:4040")

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotrust-backend")
masked_key = OPENROUTER_API_KEY[:8] + "…" if len(OPENROUTER_API_KEY) >= 8 else "****"
logger.info(f"🔐 OpenRouter API 키 확인: {masked_key}")
logger.info(f"🤖 모델: {MODEL_NAME}")

# HTTP 세션 (재사용)
http = requests.Session()
http.headers.update({"Authorization": f"Bearer {OPENROUTER_API_KEY}"})

# ────────────────────────────────────────────────────────────────────────────────
# FastAPI 초기화
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="emotrust-backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:80000",           # Expo web (필요 시)
        "exp://*",                          # Expo 개발 링크 (대략)
        "http://172.30.1.42:8000",        # 개발기기에서
        "https://d08f268191da.ngrok-free.app, # ngrok 쓰면 이거!
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ────────────────────────────────────────────────────────────────────────────────
# 모델/스키마
# ────────────────────────────────────────────────────────────────────────────────
class AnalyzeOut(BaseModel):
    emotion_score: int = Field(ge=0, le=100)
    truth_score: int = Field(ge=0, le=100)

# ────────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────────
def clamp_score(v) -> int:
    try:
        x = int(round(float(v)))
    except Exception:
        x = 0
    return max(0, min(100, x))

def extract_first_json_block(text: str) -> Optional[dict]:
    """
    모델이 코드펜스나 여분 텍스트를 섞어도 첫 번째 JSON 오브젝트를 파싱.
    """
    # 1차: 그대로 파싱 시도
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2차: 가장 간단한 { ... } 블록 탐색
    import re
    for m in re.finditer(r"\{[^{}]+\}", text, flags=re.DOTALL):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None

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

# ────────────────────────────────────────────────────────────────────────────────
# 엔드포인트
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
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
            "endpoints": {
                "root": f"{url}/",
                "analyze": f"{url}/analyze",
            },
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
            detail={
                "error": f"ngrok API 응답 코드 {resp.status_code}",
                "body": resp.text[:300],
            },
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
        "endpoints": {
            "root": f"{url}/",
            "analyze": f"{url}/analyze",
        },
    }

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(title: str = Form(...), content: str = Form(...)):
    # 프롬프트 구성
    prompt = f"""
아래는 사용자의 일기입니다. 이 글을 읽고 감정 점수와 진정성 점수를 추정하세요.

[감정 점수 기준]
- 감정 표현이 강하고 명확할수록 높은 점수 (0~100)
- 감정이 거의 드러나지 않으면 낮은 점수

[진정성 점수 기준]
- 구체적이고 사실적인 서술일수록 높은 점수 (0~100)
- 과장되거나 추상적인 내용은 낮은 점수

[제목]
{title}

[내용]
{content}

[출력 형식]
아무 설명 없이 아래와 같은 JSON만 출력하세요:
{{"emotion_score": 78, "truth_score": 92}}
""".strip()

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "너는 감정 분석 전문가야."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    # OpenRouter 권장 부가 헤더(선택)
    headers = {
        "Content-Type": "application/json",
    }
    # 필요시 참조자/타이틀 제공 (없어도 동작)
    if os.getenv("OPENROUTER_HTTP_REFERRER"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERRER")
    if os.getenv("OPENROUTER_TITLE"):
        headers["X-Title"] = os.getenv("OPENROUTER_TITLE")

    try:
        r = http.post(OPENROUTER_URL, headers=headers, json=payload, timeout=20)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="OpenRouter 응답 시간 초과")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter 호출 실패: {str(e)}")

    if r.status_code != 200:
        # OpenRouter 에러 바디 전달
        raise HTTPException(
            status_code=502,
            detail={"error": "OpenRouter 비정상 응답", "status": r.status_code, "body": r.text[:500]},
        )

    try:
        data = r.json()
        result_text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(
            status_code=502,
            detail={"error": "OpenRouter 응답 파싱 실패", "body": r.text[:500]},
        )

    # 모델 출력에서 JSON만 추출
    parsed = extract_first_json_block(result_text)
    if not parsed:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "모델 응답이 JSON 형식이 아닙니다.",
                "raw_response": result_text[:500],
            },
        )

    emotion = clamp_score(parsed.get("emotion_score"))
    truth = clamp_score(parsed.get("truth_score"))

    return AnalyzeOut(emotion_score=emotion, truth_score=truth)
