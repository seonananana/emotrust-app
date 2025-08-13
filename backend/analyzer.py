from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from acc_score import predict_with_kobert     # KoBERT 기반 정확성
from pre_score import get_lexicon             # 진정성(사전 기반)
from preproc_pii import (
    moderate_then_preprocess,
    log_moderation_event,
)


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


def pre_pipeline(
    text: str,
    denom_mode: str = "all",
    w_acc: float = 0.5,
    w_sinc: float = 0.5,
    gate: float = 0.70,
    *,
    pdf_paths: Optional[List[str]] = None,
    pdf_blobs: Optional[List[Tuple[str, bytes]]] = None,
    enable_coverage_boost: bool = True,
    coverage_boost_k: float = 0.7,
    coverage_boost_max: float = 0.15,
    min_sinc_if_no_pdf: Optional[float] = 0.40,
) -> Dict[str, Any]:

    action, clean_candidate, reasons = moderate_then_preprocess(text)

    try:
        log_moderation_event(action, reasons, clean_candidate)
    except Exception:
        pass

    if action == "block":
        return {
            "pii_action": action, "pii_reasons": reasons,
            "S_acc": 0.0, "S_sinc": 0.0, "S_pre": 0.0, "S_pre_ext": 0.0,
            "S_pre_raw": 0.0, "gate_used": normalize_gate(gate),
            "gate_used_raw": round(normalize_gate(gate) * 100, 1),
            "gate_pass": False, "tokens": 0, "matched": 0, "total": 0,
            "coverage": 0.0, "clean_text": "", "masked": False,
            "S_fact": None, "need_evidence": False,
            "claims": [], "evidence": {}
        }

    clean = clean_candidate
    masked = (action == "allow_masked")

    # ---- [3] 진정성(사전 기반) ----
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    if enable_coverage_boost:
        try:
            boost = min(coverage_boost_max, coverage_boost_k * float(cov))
            S_sinc = clamp01(S_sinc + boost)
        except Exception:
            S_sinc = clamp01(S_sinc)

    # ---- [4] KoBERT 기반 정확성 ----
    S_fact: Optional[float] = None
    claims: List[str] = []
    evidence: Dict[str, Any] = {}
    need_evidence = False

    try:
        s_acc_raw = predict_with_kobert(clean)
        S_fact = clamp01(s_acc_raw)
    except Exception:
        S_fact = None

    # ---- [5] 결합/게이트 ----
    gate_norm = normalize_gate(gate)

    if S_fact is None:
        if isinstance(min_sinc_if_no_pdf, (int, float)):
            S_sinc = max(S_sinc, clamp01(min_sinc_if_no_pdf), 0.40)
        S_pre = S_sinc
        S_pre_ext = S_pre
    else:
        denom = max(1e-9, w_acc + w_sinc)
        S_pre = (w_acc * clamp01(S_fact) + w_sinc * S_sinc) / denom
        S_pre_ext = S_pre

    gate_pass = bool(S_pre >= gate_norm)

    return {
        "pii_action": action, "pii_reasons": reasons,
        "S_acc": clamp01(0.0 if S_fact is None else S_fact),
        "S_sinc": clamp01(S_sinc), "S_pre": clamp01(S_pre),
        "S_pre_ext": clamp01(S_pre_ext),
        "S_pre_raw": round(clamp01(S_pre) * 100.0, 1),
        "gate_used": gate_norm, "gate_used_raw": round(gate_norm * 100.0, 1),
        "gate_pass": gate_pass, "tokens": int(total), "matched": int(matched),
        "total": int(total), "coverage": float(cov), "clean_text": clean,
        "masked": bool(masked), "S_fact": S_fact,
        "need_evidence": False, "claims": [], "evidence": {}
    }


async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    action, clean, _ = moderate_then_preprocess(content)
    if action == "block":
        return PreSignals(s_acc=0.0, s_sinc=0.0)

    s_acc_proxy = 0.0
    S_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc_proxy, s_sinc=S_sinc)
