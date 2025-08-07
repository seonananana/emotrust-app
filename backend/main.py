from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

# 최신 OpenAI SDK 방식: client 객체 생성
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 origin 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(title: str = Form(...), content: str = Form(...)):
    prompt = f"""제목: {title}
내용: {content}
이 글에서 감정 점수(0~100)를 추정하고, 진정성 점수(0~100)도 추정해줘.
결과는 JSON 형식으로만 응답해줘. 예시:
{{"emotion_score": 78, "truth_score": 92}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 감정 분석 전문가야."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return eval(response.choices[0].message.content)  # 단순화용, 추후 json.loads 권장
    except Exception as e:
        return {"error": f"GPT 분석 중 오류 발생: {str(e)}"}
