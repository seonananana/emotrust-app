import re
from collections import Counter

# 예시 감정 단어 사전 (NRC 사전 기반)
emotion_lexicon = {
    "기쁨": "joy", "슬픔": "sadness", "분노": "anger",
    "두려움": "fear", "놀람": "surprise", "혐오": "disgust",
    "신뢰": "trust", "기대": "anticipation"
}

def clean_text(text: str) -> str:
    # 특수문자 제거 및 소문자 변환
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text).strip().lower()

def extract_emotion_features(text: str) -> dict:
    counts = Counter()
    for word in text.split():
        if word in emotion_lexicon:
            counts[emotion_lexicon[word]] += 1
    return dict(counts)
