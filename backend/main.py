from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import os
from dotenv import load_dotenv
import openai

# 🔐 .env에서 API 키 불러오기
print("🔐 .env 파일 불러오는 중...")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"✅ API 키: {api_key}")
openai.api_key = api_key

app = FastAPI()

# 🌐 CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ 배포 시에는 특정 도메인으로 제한할 것
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📥 감정 분석 요청 처리
@app.post("/analyze")
async def analyze_emotion(
    title: str = Form(...),
    content: str = Form(...),
    file: Optional[UploadFile] = None
):
    prompt = f"""
    제목: {title}
    내용: {content}

    위 글을 감정적으로 분석해서 다음 두 가지 점수를 계산해줘.
    1. 감정 점수 (emotion_score): 감정의 강도 (0.0 ~ 1.0)
    2. 진정성 점수 (truth_score): 글이 진심으로 느껴지는 정도 (0.0 ~ 1.0)

    아래 형식의 JSON으로 정확히 반환해줘:

    {{
        "emotion_score": 0.87,
        "truth_score": 0.92
    }}
    """

    # 🔁 GPT 호출
    result = await chatgpt_emotion_analysis(prompt)
    return result

# 🤖 실제 OpenAI GPT-4 호출
async def chatgpt_emotion_analysis(prompt: str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 또는 "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        result_text = response.choices[0].message.content.strip()

        # GPT가 JSON처럼 응답한다고 가정
        return json.loads(result_text)

    except Exception as e:
        return {"error": f"GPT 분석 중 오류 발생: {str(e)}"}

@app.get("/")
def root():
    return {"message": "Hello emotrust"}