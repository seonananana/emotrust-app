from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import os

# 🔐 .env에서 API 키 불러오기
print("🔐 .env 파일 불러오는 중...")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"✅ API 키: {api_key[:8]}...")  # 보안상 일부만 출력

# OpenAI 클라이언트 객체 생성
client = OpenAI(api_key=api_key)

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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 감정 분석 전문가야."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return eval(content)  # 개발 중 편의용. 배포 전 json.loads 권장
    except Exception as e:
        return {"error": f"GPT 분석 중 오류 발생: {str(e)}"}
