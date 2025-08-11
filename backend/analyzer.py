# analyzer.py
import os
from dataclasses import dataclass
from typing import Dict, Any

from acc_score import get_kobert
from pre_score import get_lexicon
from preproc_pii import moderate_then_preprocess, log_moderation_event

# ===== 공용 유틸/컨테이너 (scoring.py 통합) =====
def clamp01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

@dataclass
class PreSignals:
    s_acc: float   # KoBERT 정확도 유사 점수 [0,1]
    s_sinc: float  # 진정성 점수 [0,1]
    def __post_init__(self):
        self.s_acc = clamp01(self.s_acc)
        self.s_sinc = clamp01(self.s_sinc)

# ===== 파이프라인 (PII → 전처리 → KoBERT → CSV) =====
def pre_pipeline(
    text: str,
    denom_mode: str = "all",
    w_acc: float = 0.5,
    w_sinc: float = 0.5,
    gate: float = 0.70,
) -> Dict[str, Any]:
    """
    [1] PII 필터 → [2] 전처리 → [3] KoBERT → [4] CSV → [5] 결합/게이트
    반환 dict에는 PII 처리 결과(action/reasons)도 포함.
    """
    # [1] PII 필터 + 마스킹/차단
    action, clean_candidate, reasons = moderate_then_preprocess(text)

    # (선택) 감사 로그
    try:
        log_moderation_event(action, reasons, clean_candidate)
    except Exception:
        pass

    if action == "block":
        return {
            "pii_action": action, "pii_reasons": reasons,
            "S_acc": 0.0, "S_sinc": 0.0, "S_pre": 0.0, "gate_pass": False,
            "tokens": 0, "matched": 0, "total": 0, "coverage": 0.0,
            "clean_text": "", "masked": False,
        }

    # [2] 전처리 완료 텍스트
    clean = clean_candidate
    masked = (action == "allow_masked")

    # [3] KoBERT 정확성
    kob = get_kobert()
    S_acc = kob.score(clean)

    # [4] CSV 진정성
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    # [5] 결합/게이트
    S_pre = w_acc * S_acc + w_sinc * S_sinc
    gate_pass = (S_pre >= gate)

    return {
        "pii_action": action, "pii_reasons": reasons,
        "S_acc": S_acc, "S_sinc": S_sinc, "S_pre": S_pre, "gate_pass": gate_pass,
        "tokens": kob.tokens_count(clean), "matched": matched, "total": total, "coverage": cov,
        "clean_text": clean, "masked": masked,
    }

# 기존 인터페이스 유지 (게이트 전 사전신호)
async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    action, clean, _ = moderate_then_preprocess(content)
    if action == "block":
        return PreSignals(s_acc=0.0, s_sinc=0.0)
    s_acc = get_kobert().score(clean)
    s_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc, s_sinc=s_sinc)
