# analyzer.py (all-in-one, integrated pipeline, PII FP/FN 개선 반영)
import os
import re
import csv
import math
import json
import uuid
import torch
import hashlib
import unicodedata
import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# =========================
# 공용 유틸/데이터 컨테이너
# =========================
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

# =========================
# PII 필터 + 전처리
# =========================
# --- PII 패턴들 ---
# 주민등록번호: 캡처 그룹 + 체크섬 검증으로 FP 감소
RRN_RE   = re.compile(r"\b(\d{6})[- ]?(\d{7})\b")

# 신용카드 후보(룬 체크로 확정)
CARD_RE  = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# 은행계좌(단순 후보) + 컨텍스트 키워드 동시 매칭 시에만 마스킹
ACC_RE   = re.compile(r"\b\d{10,14}\b")
ACC_CTX  = re.compile(
    r"(계좌|입금|송금|이체|무통장|bank|account|농협|국민|신한|우리|하나|"
    r"카카오|토스|케이뱅크|ibk|기업|수협|새마을|우체국)", re.IGNORECASE
)

# 도로명 주소 간단 패턴 + 컨텍스트 키워드 동시 매칭 시에만 마스킹
ROAD_RE  = re.compile(
    r"\b[가-힣0-9A-Za-z]+(?:로|길|대로)\s?\d+(?:-\d+)?(?:\s?\d+호|\s?\d+층)?\b"
)
ROAD_CTX = re.compile(
    r"(주소|도로명|배달|택배|배송|거주|거주지|집|사무실|건물|아파트|빌라|오피스텔|"
    r"호\b|동\b|층\b|번지|우편번호)"
)

def _luhn_ok(num_str: str) -> bool:
    """신용카드 번호 룬(Luhn) 체크: 하이픈/공백 제거 후 검증"""
    digits = [int(c) for c in re.sub(r"\D", "", num_str)]
    if not (13 <= len(digits) <= 19):
        return False
    s = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d = d * 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0

def _rrn_ok(six: str, seven: str) -> bool:
    """주민등록번호 체크섬 검증 (YYMMDD-ABCDEFG, 마지막 G는 검증숫자)"""
    try:
        nums = [int(c) for c in six + seven]
        if len(nums) != 13:
            return False
        weights = [2,3,4,5,6,7,8,9,2,3,4,5]
        s = sum(a*b for a, b in zip(nums[:12], weights))
        check = (11 - (s % 11)) % 10
        return check == nums[12]
    except Exception:
        return False

def moderate_text(text: str):
    """
    반환: (action, masked_text, reasons)
    - action: "block" | "allow" | "allow_masked"
    - reasons: 탐지 사유 코드 리스트
    정책(개선):
      * 주민등록번호: 체크섬 통과 시 block (형식만 맞으면 X)
      * 신용카드: Luhn 통과 시 block
      * 계좌번호: 숫자패턴 + 컨텍스트 동시 매칭 시 mask
      * 도로명주소: 주소패턴 + 컨텍스트 동시 매칭 시 mask
    """
    reasons: List[str] = []

    # 하드 차단: 주민등록번호(체크섬까지 통과해야 block)
    m_rrn = RRN_RE.search(text)
    if m_rrn and _rrn_ok(m_rrn.group(1), m_rrn.group(2)):
        return "block", text, ["resident_id"]

    # 하드 차단: 신용카드(룬 통과 시에만)
    for m in CARD_RE.finditer(text):
        if _luhn_ok(m.group()):
            return "block", text, ["credit_card"]

    # 마스킹 대상(컨텍스트 기반)
    masked = text
    before = masked

    # 계좌: 숫자 패턴 + 컨텍스트 키워드 동시 매칭
    if ACC_RE.search(masked) and ACC_CTX.search(masked):
        masked = ACC_RE.sub("[REDACTED:ACCOUNT]", masked)
        reasons.append("bank_account")

    # 도로명: 주소 패턴 + 컨텍스트 키워드 동시 매칭
    if ROAD_RE.search(masked) and ROAD_CTX.search(masked):
        masked = ROAD_RE.sub("[REDACTED:ROAD]", masked)
        reasons.append("road_address")

    action = "allow_masked" if masked != before else "allow"
    return action, masked, reasons

# --- 전처리 (URL/제로폭/공백/소문자/NFKC) ---
_url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def preprocess_text(text: str) -> str:
    """URL/제로폭/공백 정리 + lower + NFKC 정규화"""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _url_re.sub(" ", t)
    t = t.replace("\u200b", "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def moderate_then_preprocess(raw_text: str):
    """
    1) PII 필터 (block 또는 마스킹)
    2) 전처리 적용
    반환: (action, preprocessed_text, reasons)
    """
    action, masked, reasons = moderate_text(raw_text)
    if action == "block":
        # 그대로 반려하고, 이후 파이프라인에 넘기지 않음
        return action, "", reasons
    clean = preprocess_text(masked)
    return action, clean, reasons

# (선택) 감사 로그: 원문 저장 금지, 마스킹 텍스트 해시만 저장
def log_moderation_event(action: str, reasons: List[str], masked_text: str, sink_path: str = "moderation.log") -> str:
    event = {
        "event_id": str(uuid.uuid4()),
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "action": action,                       # "block" | "allow" | "allow_masked"
        "reasons": reasons,                     # ["resident_id", ...]
        "masked_hash": hashlib.sha256(masked_text.encode("utf-8")).hexdigest() if masked_text else "",
        "masked_preview": masked_text[:120] if masked_text else "",
    }
    try:
        with open(sink_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return event["event_id"]

# =========================
# KoBERT 임베딩 → 정확도 점수
# =========================
class KobertScorer:
    """
    - backbone: KoBERT (동결)
    - head: Linear(768->1) optional, state_dict 로드
    - 확률: sigmoid(logit / T)
    - 풀링: mean pooling(mask) 기본, "cls" 옵션 가능
    """
    def __init__(self, device: Optional[str] = None,
                 head_path: Optional[str] = None,
                 pooling: str = "mean",
                 temperature: Optional[float] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.head = None
        self.hidden_size = 768
        self.pooling = pooling  # "mean" | "cls"
        self.temperature = float(os.environ.get("KOBERT_TEMPERATURE", "1.0")) if temperature is None else float(temperature)
        self._load_models()

        if head_path and os.path.exists(head_path):
            self.head = torch.nn.Linear(self.hidden_size, 1).to(self.device)
            state = torch.load(head_path, map_location=self.device)
            self.head.load_state_dict(state)
            self.head.eval()

    def _load_models(self):
    from transformers import AutoTokenizer, AutoModel
    model_name = os.environ.get("KOBERT_MODEL", "skt/kobert-base-v1")
    self.tok = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(self.device)
    for p in self.model.parameters():  # ← 명시적 동결
        p.requires_grad = False
    self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        # last_hidden_state: [B,L,H], attention_mask: [B,L]
        mask = attention_mask.unsqueeze(-1)                 # [B,L,1]
        summed = (last_hidden_state * mask).sum(dim=1)      # [B,H]
        counts = mask.sum(dim=1).clamp(min=1)               # [B,1]
        return summed / counts

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(self.hidden_size, device=self.device)
        batch = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
            if self.pooling == "cls":
                vec = out.last_hidden_state[:, 0, :]  # [1,H]
            else:
                vec = self._mean_pool(out.last_hidden_state, batch["attention_mask"])  # [1,H]
        return vec.squeeze(0)  # [H]

    @torch.inference_mode()
    def score(self, text: str) -> float:
        vec = self.encode(text)
        if self.head is not None:
            logit = self.head(vec.unsqueeze(0)).squeeze(0)
            T = max(1e-3, float(self.temperature))
            prob = torch.sigmoid(logit / T).item()
            return clamp01(prob)

        # 임시 휴리스틱(헤드 미사용 시)
        k = float(os.environ.get("KOBERT_NORM_K", "20.0"))
        norm = torch.linalg.vector_norm(vec).item()
        prob = 1.0 - math.exp(-(norm / k))
        return clamp01(prob)

    def tokens_count(self, text: str) -> int:
        return len(self.tok.tokenize(text)) if text else 0

# 전역 싱글톤
_kobert: Optional[KobertScorer] = None
def get_kobert() -> KobertScorer:
    global _kobert
    if _kobert is None:
        head = os.environ.get("KOBERT_HEAD_PATH")  # 선택
        pooling = os.environ.get("KOBERT_POOLING", "mean")
        device = os.environ.get("KOBERT_DEVICE")   # "cpu" | "cuda"
        _kobert = KobertScorer(device=device, head_path=head, pooling=pooling)
    return _kobert

# =========================
# 진정성: 감정어 사전 기반 평균 (CSV 로딩 포함)
# =========================
# 숫자 제외: 짧은/잡음 토큰이 분모를 키우는 문제 완화
_word_re = re.compile(r"[A-Za-z가-힣]+", re.UNICODE)

class LexiconScorer:
    def __init__(self, path: str, word_col: Optional[str] = None, score_col: Optional[str] = None):
        self.vocab: Dict[str, float] = {}
        self.min_v, self.max_v = 0.0, 1.0
        self._load(path, word_col, score_col)

    def _load(self, path: str, word_col: Optional[str], score_col: Optional[str]):
        """
        CSV에서 (word, score) 추출
        - 경로: EMO_LEXICON_PATH 환경변수로 지정 권장
        - 컬럼 자동 추정: word/token/lemma/단어/용어, score/value/val/점수/가중치
        """
        with open(path, "r", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(2048)
            f.seek(0)
            try:
                dialect = sniffer.sniff(sample) if sample else csv.excel
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            cols = reader.fieldnames or []
            wcol = word_col or next((c for c in cols if c.lower() in ("word","token","lemma","단어","용어")), cols[0] if cols else "word")
            scol = score_col or next((c for c in cols if c.lower() in ("score","value","val","점수","가중치")), cols[-1] if cols else "score")
            vals = []
            for row in reader:
                if not row:
                    continue
                w = str(row.get(wcol, "")).strip().lower()
                try:
                    s = float(row.get(scol, ""))
                except Exception:
                    continue
                if w:
                    self.vocab[w] = s
                    vals.append(s)
        if vals:
            self.min_v, self.max_v = min(vals), max(vals)

    def _norm(self, x: float) -> float:
        # 사전 점수가 [0,1]이 아니면 min-max 정규화
        if self.min_v < 0.0 or self.max_v > 1.0:
            if self.max_v == self.min_v:
                return 0.0
            x = (x - self.min_v) / (self.max_v - self.min_v)
        return clamp01(x)

    def sincerity(self, text: str, mode: str = "all", alpha: float = 2.0) -> Tuple[float,int,int,float]:
        """
        returns: (S_sinc, matched_count, total_tokens, coverage)
        mode='all'    → N = 전체 단어 수 + alpha
        mode='matched'→ N = 일치 단어 수 + alpha
        alpha         → 라플라스 스무딩(짧은 글 과대평가 방지)
        """
        if not text:
            return 0.0, 0, 0, 0.0
        toks = [t.lower() for t in _word_re.findall(text)]
        if not toks:
            return 0.0, 0, 0, 0.0
        total = len(toks)
        matched_scores = [self._norm(self.vocab[t]) for t in toks if t in self.vocab]
        matched = len(matched_scores)

        N = (max(1, matched) if mode == "matched" else total) + alpha
        s = (sum(matched_scores) / N) if N > 0 else 0.0
        cov = matched / max(1, total)
        return clamp01(s), matched, total, cov

# 전역 사전 로더 (CSV 경로 환경변수 지원)
_lex: Optional[LexiconScorer] = None
def get_lexicon() -> LexiconScorer:
    global _lex
    if _lex is None:
        path = os.environ.get("EMO_LEXICON_PATH", "/mnt/data/nrc_words.csv")
        wcol = os.environ.get("EMO_LEX_WORD_COL")
        scol = os.environ.get("EMO_LEX_SCORE_COL")
        _lex = LexiconScorer(path, wcol, scol)
    return _lex

# =========================
# 파이프라인 (PII → 전처리 → KoBERT → CSV)
# =========================
def pre_pipeline(text: str, denom_mode: str = "all",
                 w_acc: float = 0.5, w_sinc: float = 0.5,
                 gate: float = 0.70):
    """
    [1] PII 필터 → [2] 전처리 → [3] KoBERT → [4] CSV → [5] 결합/게이트
    반환 dict에는 PII 처리 결과(action/reasons)도 포함.
    """
    # [1] PII 필터 + 마스킹/차단 + 전처리
    action, clean_candidate, reasons = moderate_then_preprocess(text)

    # (선택) 감사 로그 남기기 — 원문 저장 금지, 마스킹 텍스트만
    try:
        log_moderation_event(action, reasons, clean_candidate)
    except Exception:
        pass

    if action == "block":
        return {
            "pii_action": action,            # "block"
            "pii_reasons": reasons,          # ["resident_id"] 등
            "S_acc": 0.0, "S_sinc": 0.0,
            "S_pre": 0.0, "gate_pass": False,
            "tokens": 0, "matched": 0, "total": 0, "coverage": 0.0,
            "clean_text": "", "masked": False,
        }

    # [2] 전처리까지 완료된 텍스트
    clean = clean_candidate
    masked = (action == "allow_masked")

    # [3] KoBERT
    kob = get_kobert()
    S_acc = kob.score(clean)

    # [4] CSV 진정성
    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    # [5] 결합/게이트
    S_pre = w_acc * S_acc + w_sinc * S_sinc
    gate_pass = (S_pre >= gate)

    return {
        "pii_action": action,               # "allow" | "allow_masked"
        "pii_reasons": reasons,             # 마스킹 이유 코드 목록(없으면 [])
        "S_acc": S_acc, "S_sinc": S_sinc,
        "S_pre": S_pre, "gate_pass": gate_pass,
        "tokens": kob.tokens_count(clean),
        "matched": matched, "total": total, "coverage": cov,
        "clean_text": clean,
        "masked": masked,
    }

# 기존 인터페이스 유지용 (게이트 전 사전신호)
async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    action, clean, reasons = moderate_then_preprocess(content)
    if action == "block":
        return PreSignals(s_acc=0.0, s_sinc=0.0)
    s_acc = get_kobert().score(clean)
    s_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc, s_sinc=s_sinc)
