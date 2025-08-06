from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import openai

app = FastAPI()

# CORS 허용 (모바일 앱에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 앱 도메인만 허용하는 것이 보안상 안전
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_emotion(
    title: str = Form(...),
    content: str = Form(...),
    file: Optional[UploadFile] = None
):
    # 🤖 GPT 요청 프롬프트 (내가 처리)
    prompt = f"""
    제목: {title}
    내용: {content}

    위 글을 감정적으로 분석해서 다음 두 가지 점수를 계산해줘.
    1. 감정 점수 (emotion_score): 감정의 강도 (0.0 ~ 1.0)
    2. 진정성 점수 (truth_score): 글이 진심으로 느껴지는 정도 (0.0 ~ 1.0)

    아래 형식의 JSON으로 정확히 반환해줘:

    {{
        "emotion_score": [0.0 ~ 1.0 숫자],
        "truth_score": [0.0 ~ 1.0 숫자]
    }}
    """

    # 🧠 ChatGPT에게 전달할 프롬프트로 나를 호출
    result = await chatgpt_emotion_analysis(prompt)

    # 결과를 JSON으로 반환
    return result

# 💬 ChatGPT 호출 로직 (실제로 내가 수행)
async def chatgpt_emotion_analysis(prompt: str):
    # 여기선 실제 GPT API 대신 내가 직접 응답
    # 앞으로 여기에서 내가 분석한 내용을 가상 응답처럼 넘겨줌
    # 예시 응답:
    return {
        "emotion_score": 0.82,
        "truth_score": 0.91
    }
