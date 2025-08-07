from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
from dotenv import load_dotenv

# 🔐 .env에서 API 키 불러오기
print("🔐 .env 파일 불러오는 중...")
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("❌ .env에서 OPENROUTER_API_KEY를 불러오지 못했습니다.")
print(f"✅ API 키 앞자리: {api_key[:8]}...")

# FastAPI 앱 초기화
app = FastAPI()

# CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 분석 요청 처리
@app.post("/analyze")
async def analyze(title: str = Form(...), content: str = Form(...)):
    prompt = f"""
    제목: {title}
    내용: {content}

    이 글을 읽고 아래 기준으로 감정 점수와 진정성 점수를 각각 추정해줘.

    [감정 점수 기준]
    - 감정 표현이 강하고 명확할수록 높은 점수
    - 감정이 거의 드러나지 않으면 낮은 점수

    [진정성 점수 기준]
    - 내용이 사실적이고 구체적일수록 높은 점수
    - 내용이 모호하거나 과장되어 있으면 낮은 점수

    결과는 반드시 JSON 형식으로만 응답해줘. 예시:
    {{"emotion_score": 78, "truth_score": 92}}
    """

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mixtral-8x7b-instruct",
                "messages": [
                    {"role": "system", "content": "너는 감정 분석 전문가야."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
        )

        response.raise_for_status()
        data = response.json()
        result_text = data["choices"][0]["message"]["content"]
        print("🧠 모델 응답 내용:", result_text)

        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            return {
                "error": "❌ 모델 응답이 JSON 형식이 아닙니다",
                "raw_response": result_text
            }

    except Exception as e:
        return {"error": f"OpenRouter 분석 중 오류 발생: {str(e)}"}
