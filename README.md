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

## 모델 다운로드

학습된 KoBERT 모델(`kobert_regression.pt`)은 아래 링크에서 다운로드할 수 있습니다:
👉 [Google Drive 링크](https://drive.google.com/file/d/1AKTZDQAEtLW9OHQA9hH5rGyYPJdcZcKb/view?usp=share_link)
> 다운로드 후 `backend/data/kobert_regression.pt`에 위치시켜 주세요.
