# main.py
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
import requests
import logging
import os
import json
import re
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────────
# 환경 / 로깅
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

def as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("emotrust-backend")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
MODEL_NAME = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct").strip()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "64"))
REQUEST_TIMEOUT_SECS = float(os.getenv("REQUEST_TIMEOUT_SECS", "20"))

# 프런트 CORS
ALLOW_ORIGINS_ENV = os.getenv("ALLOW_ORIGINS") or "*"
ALLOW_ORIGINS = [o.strip() for o in ALLOW_ORIGINS_ENV.split(",") if o.strip()]

# ngrok/도메인
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or os.getenv("NGROK_PUBLIC_URL") or "").strip() or None
NGROK_API_URL = os.getenv("NGROK_API_URL", "http://127.0.0.1:4040").strip()

# 폴백 정책
FALLBACK_ON_402 = as_bool(os.getenv("FALLBACK_ON_402", "1"), True)
FALLBACK_ON_ERROR = as_bool(os.getenv("FALLBACK_ON_ERROR", "1"), True)

USE_OPENROUTER = bool(OPENROUTER_API_KEY)
masked_key = (OPENROUTER_API_KEY[:8] + "…") if USE_OPENROUTER else "(none)"
logger.info(f"🔐 OpenRouter API 키: {masked_key}")
logger.info(f"🤖 모델: {MODEL_NAME}  |  max_tokens={MAX_TOKENS}  |  use_openrouter={USE_OPENROUTER}")

# HTTP 세션 (OpenRouter용; 키 없으면 생성 안 함)
http = None
if USE_OPENROUTER:
    http = requests.Session()
    http.headers.update({"Authorization": f"Bearer {OPENROUTER_API_KEY}"})


# ────────────────────────────────────────────────────────────────────────────────
# FastAPI 초기화 + CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="emotrust-backend", version="1.1.0")

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
    """모델이 코드펜스/부가 텍스트를 섞어도 첫 JSON 오브젝트를 파싱."""
    # 1) 그대로 파싱
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) ```json ... ``` 또는 ``` ... ``` 내부 추출
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            pass
    # 3) 가장 단순한 { ... } 블록
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

def heuristic_scores(title: str, content: str):
    """크레딧 부족/오류 시 임시 점수 계산(데모/개발용). 필요에 맞게 조정 가능."""
    text = f"{title}\n{content}"
    emo_words = ['너무','정말','진짜','완전','매우','엄청','대단히','굉장히','화가','기쁘','슬프','불안','긴장','짜증','행복','후회','억울']
    emo_hits = sum(text.count(w) for w in emo_words)
    exclam = text.count('!') + text.count('😭') + text.count('ㅠ') + text.count('ㅜ')
    emotion = min(100, 30 + emo_hits * 10 + exclam * 5)

    digits = len(re.findall(r'\d', text))
    units = ['시','분','월','일','원','만원','km','킬로','kg','개','명','병원','회사','학교']
    unit_hits = sum(text.count(u) for u in units)
    quotes = text.count('"') + text.count('“') + text.count('”')
    length_bonus = min(30, len(text) // 80)
    truth = min(100, 30 + digits * 3 + unit_hits * 4 + quotes * 2 + length_bonus)

    return int(emotion), int(truth)


# ────────────────────────────────────────────────────────────────────────────────
# 엔드포인트
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "use_openrouter": USE_OPENROUTER,
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

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(title: str = Form(...), content: str = Form(...)):
    # OpenRouter를 쓰지 않는 모드(키 없음/비활성) → 즉시 폴백
    if not USE_OPENROUTER:
        e, t = heuristic_scores(title, content)
        return AnalyzeOut(emotion_score=e, truth_score=t)

    # 프롬프트
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
        "max_tokens": MAX_TOKENS,
    }

    headers = {"Content-Type": "application/json"}
    if os.getenv("OPENROUTER_HTTP_REFERRER"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERRER")
    if os.getenv("OPENROUTER_TITLE"):
        headers["X-Title"] = os.getenv("OPENROUTER_TITLE")

    # 호출
    try:
        r = http.post(OPENROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECS)
    except requests.Timeout:
        if FALLBACK_ON_ERROR:
            e, t = heuristic_scores(title, content)
            return AnalyzeOut(emotion_score=e, truth_score=t)
        raise HTTPException(status_code=504, detail="OpenRouter 응답 시간 초과")
    except Exception as e:
        if FALLBACK_ON_ERROR:
            e_, t_ = heuristic_scores(title, content)
            return AnalyzeOut(emotion_score=e_, truth_score=t_)
        raise HTTPException(status_code=502, detail=f"OpenRouter 호출 실패: {str(e)}")

    # 402 → 크레딧 부족 폴백
    if r.status_code == 402 and FALLBACK_ON_402:
        e, t = heuristic_scores(title, content)
        return AnalyzeOut(emotion_score=e, truth_score=t)

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"error": "OpenRouter 비정상 응답", "status": r.status_code, "body": r.text[:500]},
        )

    # 파싱
    try:
        data = r.json()
        result_text = data["choices"][0]["message"]["content"]
    except Exception:
        if FALLBACK_ON_ERROR:
            e, t = heuristic_scores(title, content)
            return AnalyzeOut(emotion_score=e, truth_score=t)
        raise HTTPException(
            status_code=502,
            detail={"error": "OpenRouter 응답 파싱 실패", "body": r.text[:500]},
        )

    parsed = extract_first_json_block(result_text)
    if not parsed:
        if FALLBACK_ON_ERROR:
            e, t = heuristic_scores(title, content)
            return AnalyzeOut(emotion_score=e, truth_score=t)
        raise HTTPException(
            status_code=502,
            detail={"error": "모델 응답이 JSON 형식이 아닙니다.", "raw_response": result_text[:500]},
        )

    emotion = clamp_score(parsed.get("emotion_score"))
    truth = clamp_score(parsed.get("truth_score"))
    return AnalyzeOut(emotion_score=emotion, truth_score=truth)
