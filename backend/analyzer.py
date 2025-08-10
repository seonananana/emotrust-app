# analyzer.py (all-in-one)
import os, math, re, csv, unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# =========================
# 공용 유틸/데이터 컨테이너
# =========================
def clamp01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

@dataclass
class PreSignals:
    s_acc: float   # KoBERT 정확도 유사 점수 [0,1]
    s_sinc: float  # 진정성 점수 [0,1]
    def __post_init__(self):
        self.s_acc = clamp01(self.s_acc)
        self.s_sinc = clamp01(self.s_sinc)

# =========================
# 전처리
# =========================
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

# =========================
# KoBERT 임베딩 → 정확도 점수
# =========================
import torch

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
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
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
                vec = self._mean_pool(out.last_hidden_state, batch["attention_mask"])
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
            dialect = sniffer.sniff(sample) if sample else csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            cols = reader.fieldnames or []
            wcol = word_col or next((c for c in cols if c.lower() in ("word","token","lemma","단어","용어")), cols[0])
            scol = score_col or next((c for c in cols if c.lower() in ("score","value","val","점수","가중치")), cols[-1])
            vals = []
            for row in reader:
                w = str(row[wcol]).strip().lower()
                try:
                    s = float(row[scol])
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
# 파이프라인 (전처리 + KoBERT + CSV사전)
# =========================
def pre_pipeline(text: str, denom_mode: str = "all",
                 w_acc: float = 0.5, w_sinc: float = 0.5,
                 gate: float = 0.70):
    """
    반환 dict에는 디버깅/로깅용 지표 포함.
    """
    clean = preprocess_text(text)

    kob = get_kobert()
    S_acc = kob.score(clean)

    lex = get_lexicon()
    S_sinc, matched, total, cov = lex.sincerity(clean, mode=denom_mode)

    S_pre = w_acc * S_acc + w_sinc * S_sinc
    gate_pass = (S_pre >= gate)

    return {
        "S_acc": S_acc,
        "S_sinc": S_sinc,
        "S_pre": S_pre,
        "gate_pass": gate_pass,
        "tokens": kob.tokens_count(clean),
        "matched": matched,
        "total": total,
        "coverage": cov,
        "clean_text": clean,
    }

# 기존 인터페이스 유지용
async def build_pre_signals(content: str, denom_mode: str = "all") -> PreSignals:
    clean = preprocess_text(content)
    s_acc = get_kobert().score(clean)
    s_sinc, _, _, _ = get_lexicon().sincerity(clean, mode=denom_mode)
    return PreSignals(s_acc=s_acc, s_sinc=s_sinc)
