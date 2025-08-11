# analyzer.py (B안 전용: PDF 팩트체크 + 사전 진정성)
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from acc_score import score_with_pdf         # B안 메인
from pre_score import get_lexicon            # 진정성(사전) 점수
from preproc_pii import moderate_then_preprocess, log_moderation_event  # PII + 전처리

# ===== 공용 유틸/컨테이너 =====
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
    # 하위호환을 위해 이름만 유지: s_acc 자리에 S_fact를 매핑하거나, 증거 없으면 0.0/0.5 등 중립값
    s_acc: float   # 이제 KoBERT가 아니라 S_fact(가능하면) 또는 중립값
    s_sinc: float  # 진정성 점수 [0,1]
    def __post_init__(self):
        self.s_acc = clamp01(self.s_acc)
        self.s_sinc = clamp01(self.s_sinc)

# ===== 파이프라인 (PII → 전처리 → 사전 → PDF팩트체크) =====
def pre_pipeline(
    text: str,
    denom_mode: str = "all",
    w_acc: float = 0.5,    # ← 이름만 유지, 이제 S_fact 가중치로 사용
    w_sinc: float = 0.5,
    gate: float = 0.70,
    *,
    pdf_paths: Optional[List[str]] = None,   # PDF 없으면 검증 불가 플래그만 반환
) -> Dict[str, Any]:
    """
    [1] PII 필터 → [2] 전처리 → [3] 진정성(CSV) → [4] PDF팩트체크(B안) → [5] 결합/게이트
    반환: 기본 지표 + (있으면) PDF 사실성 결과
    """
    # [1] PII 필터 + 전처리
    action, clean_candidate, reasons = moderate_then_preprocess(text)

    # 감사 로그(선택)
    try:
        log_moderation_event(action, reasons, clean_candidate)
    except Exception:
        pass

    if action == "block":
        return {
            "pii_action": action, "pii_reasons": reasons,
            "S_acc": 0.0, "S_sinc": 0.0, "S_pre": 0.0, "S_pre_ext": 0.0,
            "gate_pass": False,
            "tokens": 0, "matched": 0, "total": 0, "coverage": 0.0,
            "clean_text": "", "masked": False,
            "S_fact": None, "need_evidence": False, "claims": [], "evidence": {},
        }

    # [2] 전처리 완료 텍스트
    clean = clean_candidate
    masked = (action == "allow_masked")

    # [3] 진정성(사전)
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    # [4] PDF 기반 사실성(B안)
    S_fact = None
    need_evidence = False
    claims: List[str] = []
    evidence: Dict[str, Any] = {}
    try:
        fc = score_with_pdf(clean_text=clean, pdf_paths=pdf_paths)
        S_fact = fc.get("S_fact")
        need_evidence = bool(fc.get("need_evidence"))
        claims = list(fc.get("claims") or [])
        evidence = dict(fc.get("evidence") or {})
    except Exception:
        S_fact = None
        need_evidence = False
        claims = []
        evidence = {}

    # [5] 결합/게이트
    # - PDF가 없거나 검증 불가면(S_fact=None) S_sinc만으로 S_pre를 계산(가중치 자동 정규화)
    if S_fact is None:
        S_pre = (w_sinc * S_sinc) / max(1e-9, w_sinc)
        S_pre_ext = S_pre
    else:
        denom = max(1e-9, w_acc + w_sinc)
        S_pre = (w_acc * clamp01(S_fact) + w_sinc * S_sinc) / denom
        S_pre_ext = S_pre  # 확장 점수 필드 유지(필요시 별도 가중 추가 가능)

    gate_pass = (S_pre >= gate)

    # 하위호환: S_acc 필드에 S_fact를 매핑(없으면 0.0)
    return {
        "pii_action": action, "pii_reasons": reasons,
        "S_acc": clamp01(0.0 if S_fact is None else S_fact),  # 이름만 유지
        "S_sinc": S_sinc,
        "S_pre": S_pre, "S_pre_ext": S_pre_ext,
        "gate_pass": gate_pass,
        "tokens": total,               # KoBERT 없음 → 전체 토큰 수로 대체(원하면 분리)
        "matched": matched, "total": total, "coverage": cov,
        "clean_text": clean, "masked": masked,
        # B안 결과
        "S_fact": S_fact, "need_evidence": need_evidence,
        "claims": claims, "evidence": evidence,
    }

# 하위호환용(가급적 pre_pipeline 직접 호출 권장)
async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    action, clean, _ = moderate_then_preprocess(content)
    if action == "block":
        return PreSignals(s_acc=0.0, s_sinc=0.0)
    # PDF 없이는 사실성 산출이 불가 → 중립값(0.0 또는 0.5) 중 택1
    # 운영에서 보수적으로 가려면 0.0, 중립적이면 0.5
    s_acc_proxy = 0.0
    s_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc_proxy, s_sinc=s_sinc)
