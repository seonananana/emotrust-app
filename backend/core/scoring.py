# backend/core/scoring.py

from core.model import predict_s_acc
from utils.preprocess import preprocess_text
from pre_score import get_lexicon

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def normalize_gate(g: float, default: float = 0.70) -> float:
    try:
        g = float(g)
    except Exception:
        return default
    if g <= 0:
        return 0.0
    if 1.0 < g <= 100.0:
        return g / 100.0
    return g

def run_scoring_pipeline(
    text: str,
    denom_mode: str = "all",
    w_acc: float = 0.5,
    w_sinc: float = 0.5,
    gate: float = 0.7
):
    clean = preprocess_text(text)
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    S_acc = predict_s_acc(clean)

    denom = max(1e-9, w_acc + w_sinc)
    S_pre = (w_acc * S_acc + w_sinc * S_sinc) / denom

    gate_norm = normalize_gate(gate)
    gate_pass = S_pre >= gate_norm

    return {
        "S_acc": round(S_acc, 3),
        "S_sinc": round(S_sinc, 3),
        "S_pre": round(S_pre, 3),
        "S_pre_raw": round(S_pre * 100, 1),
        "gate_used": gate_norm,
        "gate_pass": gate_pass,
        "matched": matched,
        "total": total,
        "coverage": cov,
    }

def _score_extras_with_comments(sc, meta):
    return {"S_effective": sc.get("S_pre", 0.0)}
