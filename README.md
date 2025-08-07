# emotrust-app

감정 기반 신뢰 회복 플랫폼  
GPT-4를 이용한 감정·진정성 분석 + 블록체인/토큰 시스템 도입 예정

---
## 📦 프로젝트 구조
emotrust-app/
├── frontend/ # Expo 기반 모바일 앱 (React Native)
├── backend/ # FastAPI 백엔드 (OpenAI GPT 분석)

## 🚀 실행 방법

### ✅ 백엔드 실행

```bash
cd backend
uvicorn main:app --reload
.env 파일 필요:
OPENAI_API_KEY=your_openai_api_key
