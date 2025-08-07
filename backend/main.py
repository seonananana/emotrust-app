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
