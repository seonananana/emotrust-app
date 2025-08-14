from utils.preprocess import extract_emotion_features
from core.model import score_accuracy

def pre_pipeline(text: str, mode: str = "all"):
    # 감정 기반 진정성 점수 계산
    features = extract_emotion_features(text)
    if mode == "pos":
        selected = {k: v for k, v in features.items() if k in ["trust", "joy", "anticipation", "surprise"]}
    elif mode == "neg":
        selected = {k: v for k, v in features.items() if k in ["fear", "anger", "sadness", "disgust"]}
    else:
        selected = features
    denom = sum(selected.values()) or 1e-5
    s_sinc = sum(selected.values()) / denom
    return s_sinc, denom

def combine_scores(s_sinc: float, s_acc: float, w_sinc: float = 0.5, w_acc: float = 0.5):
    return round(w_sinc * s_sinc + w_acc * s_acc, 4)
