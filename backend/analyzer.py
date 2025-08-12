# analyzer.py  — B안 전용: PDF 팩트체크 + 사전 진정성
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from acc_score import score_with_pdf          # B안 메인 (PDF 기반 팩트 스코어러)
from pre_score import get_lexicon             # 진정성(사전) 점수 계산기
from preproc_pii import (
    moderate_then_preprocess,
    log_moderation_event,
)  # PII + 전처리


# =============================
# 공용 유틸
# =============================
def clamp01(x) -> float:
    """값을 [0,1] 범위로 클램프."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_gate(g: float, default: float = 0.70) -> float:
    """
    프론트에서 0~1 또는 0~100 어느 스케일로 보내도 0~1로 정규화.
    - g <= 0: 0.0
    - 1 < g <= 100: 100점 만점 스케일로 가정하여 /100
    - 그 외: 0~1로 가정
    """
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
    """
    하위호환 유지용 컨테이너.
    - s_acc: 예전 KoBERT 정확도 자리에 현재는 S_fact(가능하면) 또는 중립값을 매핑
    - s_sinc: 사전 기반 진정성 점수
    """
    s_acc: float   # 현재 의미: S_fact 또는 중립값
    s_sinc: float  # [0,1]

    def __post_init__(self):
        self.s_acc = clamp01(self.s_acc)
        self.s_sinc = clamp01(self.s_sinc)


# =============================
# 파이프라인
# =============================
def pre_pipeline(
    text: str,
    denom_mode: str = "all",
    w_acc: float = 0.5,     # 이름만 유지: 현재는 S_fact 가중치
    w_sinc: float = 0.5,
    gate: float = 0.70,
    *,
    pdf_paths: Optional[List[str]] = None,   # PDF 없으면 검증 불가 플래그만 반환
    # --- 옵션 (운영 중 튜닝을 쉽게 하려면 ENV로 매핑해도 됨) ---
    enable_coverage_boost: bool = True,
    coverage_boost_k: float = 0.5,           # S_sinc += min(max_boost, k * coverage)
    coverage_boost_max: float = 0.10,        # 최대 가산치 (예: 0.1 → 10점)
    min_sinc_if_no_pdf: Optional[float] = 0.20,  # PDF 없음/검증불가일 때 S_sinc 바닥값(비활성화는 None)
) -> Dict[str, Any]:
    """
    [1] PII 필터 → [2] 전처리 → [3] 진정성(CSV) → [4] PDF 팩트체크 → [5] 결합/게이트
    반환: 기본 지표 + (있으면) PDF 정확성 결과

    스케일:
      - 모든 내부 스코어는 0~1.
      - 응답에 보기용 0~100 원점수(S_pre_raw, gate_used_raw) 동시 제공.
      - gate 인자는 0~1 또는 0~100 어느 쪽이든 허용(자동 정규화).
    """
    # ---- [1] PII 필터 + 전처리 ----
    action, clean_candidate, reasons = moderate_then_preprocess(text)

    # 감사 로그 (실패해도 전체 파이프라인은 계속)
    try:
        log_moderation_event(action, reasons, clean_candidate)
    except Exception:
        pass

    if action == "block":
        # 차단: 완전 기본값으로 반환
        return {
            "pii_action": action, "pii_reasons": reasons,
            "S_acc": 0.0, "S_sinc": 0.0, "S_pre": 0.0, "S_pre_ext": 0.0,
            "S_pre_raw": 0.0,
            "gate_used": normalize_gate(gate), "gate_used_raw": round(normalize_gate(gate) * 100, 1),
            "gate_pass": False,
            "tokens": 0, "matched": 0, "total": 0, "coverage": 0.0,
            "clean_text": "", "masked": False,
            "S_fact": None, "need_evidence": False, "claims": [], "evidence": {},
        }

    # ---- [2] 전처리 완료 텍스트 ----
    clean = clean_candidate
    masked = (action == "allow_masked")

    # ---- [3] 진정성(사전) ----
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)  # cov∈[0,1]

    # 선택: 커버리지 가산점 (상한 coverage_boost_max)
    if enable_coverage_boost:
        try:
            boost = min(coverage_boost_max, coverage_boost_k * float(cov))
            S_sinc = clamp01(S_sinc + boost)
        except Exception:
            # 실패 시 원값 유지
            S_sinc = clamp01(S_sinc)

    # ---- [4] PDF 기반 정확성 (B안) ----
    S_fact: Optional[float] = None
    need_evidence = False
    claims: List[str] = []
    evidence: Dict[str, Any] = {}
    try:
        fc = score_with_pdf(clean_text=clean, pdf_paths=pdf_paths)
        # 다양한 키명 호환
        s_acc_raw = None
        if isinstance(fc, dict):
            s_acc_raw = fc.get("S_fact", None)
            if s_acc_raw is None:
                s_acc_raw = fc.get("S_acc", None)
            if s_acc_raw is None:
                s_acc_raw = fc.get("S_acc_pdf", None)
            need_evidence = bool(fc.get("need_evidence", False))
            claims = list(fc.get("claims") or [])
            evidence = dict(fc.get("evidence") or {})
        S_fact = None if s_acc_raw is None else clamp01(s_acc_raw)
    except Exception:
        S_fact = None
        need_evidence = False
        claims = []
        evidence = {}

    # ---- [5] 결합/게이트 ----
gate_norm = normalize_gate(gate)

if S_fact is None:
    # PDF 없음/검증 불가 → S_sinc만으로 계산
    if isinstance(min_sinc_if_no_pdf, (int, float)):
        # 바닥값 0.40 보장 + 사용자 최솟값 반영
        S_sinc = max(S_sinc, 0.40)
        S_sinc = max(S_sinc, clamp01(min_sinc_if_no_pdf))
    S_pre = (w_sinc * S_sinc) / max(1e-9, w_sinc)
    S_pre_ext = S_pre
else:
    denom = max(1e-9, w_acc + w_sinc)
    S_pre = (w_acc * clamp01(S_fact) + w_sinc * S_sinc) / denom
    S_pre_ext = S_pre

gate_pass = bool(S_pre >= gate_norm)

    # 하위호환 필드 주석:
    # - S_acc : 과거 '정확도' 키를 기대하는 소비자(프론트/DB)를 위해 유지. 의미는 S_fact 또는 0.0.
    return {
        "pii_action": action, "pii_reasons": reasons,

        "S_acc": clamp01(0.0 if S_fact is None else S_fact),  # 하위호환 이름
        "S_sinc": clamp01(S_sinc),
        "S_pre": clamp01(S_pre),
        "S_pre_ext": clamp01(S_pre_ext),

        # 보기용 원점수(0~100) 및 실제 사용한 게이트(둘 다 제공)
        "S_pre_raw": round(clamp01(S_pre) * 100.0, 1),
        "gate_used": gate_norm,
        "gate_used_raw": round(gate_norm * 100.0, 1),

        "gate_pass": gate_pass,

        # 토큰/커버리지
        "tokens": int(total),
        "matched": int(matched),
        "total": int(total),
        "coverage": float(cov),

        "clean_text": clean,
        "masked": bool(masked),

        # B안 결과
        "S_fact": None if S_fact is None else float(S_fact),
        "need_evidence": bool(need_evidence),
        "claims": claims,
        "evidence": evidence,
    }


# =============================
# 하위호환용 API (가능하면 pre_pipeline 직접 호출)
# =============================
async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    """
    KoBERT 없는 B안에서의 간단 신호 생성:
      - PDF 정확도는 없으므로 s_acc는 중립값(기본 0.0)
      - 사전 기반 진정성만 계산
    """
    action, clean, _ = moderate_then_preprocess(content)
    if action == "block":
        return PreSignals(s_acc=0.0, s_sinc=0.0)

    s_acc_proxy = 0.0  # 중립값(정책에 따라 0.0 또는 0.5 가능)
    s_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc_proxy, s_sinc=s_sinc)
