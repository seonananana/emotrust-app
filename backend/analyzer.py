from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from pre_score import get_lexicon             # 진정성(사전 기반)
from preproc_pii import preprocess_text

# ---------------------------
# KoBERT 회귀 모델 정의
# ---------------------------
class KoBERTRegressor(nn.Module):
    def __init__(self):
        super(KoBERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.regressor(cls_output).squeeze(1)

# ---------------------------
# 정확성 점수 예측 함수
# ---------------------------
def predict_s_acc(text: str) -> float:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

    model = KoBERTRegressor().to(DEVICE)
    model.load_state_dict(torch.load("kobert_regression.pt", map_location=DEVICE))
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=64
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        score = model(input_ids, attention_mask).item()
        return max(0.0, min(1.0, score))


def clamp01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return min(1.0, max(0.0, x))

def normalize_gate(g: float, default: float = 0.70) -> float:
    try:
        g = float(g)
    except Exception:
        return clamp01(default)
    if g <= 0:
        return 0.0
    if 1.0 < g <= 100.0:
        return clamp01(g / 100.0)
    return clamp01(g)

@dataclass
class PreSignals:
    s_acc: float
    s_sinc: float

    def __post_init__(self):
        self.s_acc = clamp01(self.s_acc)
        self.s_sinc = clamp01(self.s_sinc)

from preproc_pii import preprocess_text

# ...

def pre_pipeline(
    text: str,
    denom_mode: str = "all",
    w_acc: float = 0.5,
    w_sinc: float = 0.5,
    gate: float = 0.3,
    *,
    enable_coverage_boost: bool = True,
    coverage_boost_k: float = 0.7,
    coverage_boost_max: float = 0.15,
) -> Dict[str, Any]:

    clean = preprocess_text(text)
    masked = False
    reasons = []

    # 진정성 계산
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    if enable_coverage_boost:
        try:
            boost = min(coverage_boost_max, coverage_boost_k * float(cov))
            S_sinc = clamp01(S_sinc + boost)
        except Exception:
            S_sinc = clamp01(S_sinc)

    # 정확성 계산
    S_fact: Optional[float] = None
    try:
        s_acc_raw = predict_s_acc(clean)
        S_fact = clamp01(s_acc_raw)
    except Exception:
        S_fact = None

    # S_pre 계산
    denom = max(1e-9, w_acc + w_sinc)
    S_pre = (w_acc * clamp01(S_fact or 0.0) + w_sinc * S_sinc) / denom
    S_pre_ext = S_pre

    gate_norm = normalize_gate(gate)
    gate_pass = bool(S_pre >= gate_norm)

    return {
        "S_acc": clamp01(0.0 if S_fact is None else S_fact),
        "S_sinc": clamp01(S_sinc), "S_pre": clamp01(S_pre),
        "S_pre_ext": clamp01(S_pre_ext),
        "S_pre_raw": round(clamp01(S_pre) * 100.0, 1),
        "gate_used": gate_norm, "gate_used_raw": round(gate_norm * 100.0, 1),
        "gate_pass": gate_pass, "tokens": int(total), "matched": int(matched),
        "total": int(total), "coverage": float(cov), "clean_text": clean,
        "masked": bool(masked), "S_fact": S_fact,
        "need_evidence": False, "claims": [], "evidence": {},
        "pii_action": "allow", "pii_reasons": []  # 남겨도 무방 (빈 값)
    }
    
async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    clean = preprocess_text(content)
    s_acc_proxy = 0.0
    S_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc_proxy, s_sinc=S_sinc)
