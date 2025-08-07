from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv

# 🔐 .env에서 API 키 불러오기
print("🔐 .env 파일 불러오는 중...")
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")  # ✅ 키 이름도 변경
print(f"✅ API 키: {api_key[:8]}...")  # 보안상 일부만 출력

# FastAPI 앱 초기화
app = FastAPI()

# CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 특정 origin으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 분석 요청 처리
@app.post("/analyze")
async def analyze(title: str = Form(...), content: str = Form(...)):
    prompt = f"""제목: {title}
내용: {content}
이 글에서 감정 점수(0~100)를 추정하고, 진정성 점수(0~100)도 추정해줘.
결과는 JSON 형식으로만 응답해줘. 예시:
{{"emotion_score": 78, "truth_score": 92}}"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mixtral-8x7b-instruct",  # ✅ 모델명 확정
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
        return eval(result_text)  # 개발 중 편의용. 실서비스는 json.loads 권장

    except Exception as e:
        return {"error": f"OpenRouter 분석 중 오류 발생: {str(e)}"}
