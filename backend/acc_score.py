# acc_score.py
# B안: PDF 기반 팩트체크 (Claim → Retrieval → NLI) + 증거 없으면 업로드 유도
# - 선택 의존성(있으면 품질↑): pymupdf, sentence-transformers, transformers
# - 의존성 없을 때는 보워드(BoW) 유사도/근사 NLI로 폴백

import os
import re
import math
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# =========================
# 내부 유틸
# =========================
def clamp01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def _normalize_spaces(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# 선택 의존성 체크
# =========================
_HAS_PYMUPDF = False
_HAS_SENT_EMB = False
_HAS_NLI = False
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENT_EMB = True
except Exception:
    pass

try:
    from transformers import AutoTokenizer as _NLI_Tok, AutoModelForSequenceClassification as _NLI_Model
    _HAS_NLI = True
except Exception:
    pass

# =========================
# 주장(Claim) 추출
# =========================
def _split_sentences_ko(text: str) -> List[str]:
    """간이 문장 분리 (정교 분리는 kss 등으로 교체 가능)"""
    if not text:
        return []
    s = re.sub(r"([\.!?])", r"\1\n", text)
    parts = [p.strip() for p in s.splitlines() if p.strip()]
    return parts

def _is_claim_like(sent: str) -> bool:
    if not sent or len(sent) < 12:
        return False
    if re.search(r"\d{4}년|\d{1,2}월|\d{1,2}일|[\d,]+|%|억원|만명|천명|km|kg", sent):
        return True
    if re.search(r"(이다|였다|한다|됩니다|임\.?|라고 하|로 밝혀)", sent):
        return True
    return False

def extract_claims(article_text: str, max_claims: int = 3) -> List[str]:
    sents = _split_sentences_ko(article_text)
    cands = [s for s in sents if _is_claim_like(s)]

    def _score_claim(s: str) -> float:
        num_ratio = len(re.findall(r"\d", s)) / max(1, len(s))
        return 0.7 * num_ratio + 0.3 * min(len(s), 200) / 200.0

    cands.sort(key=_score_claim, reverse=True)
    return cands[:max_claims]

# =========================
# 임베더 (SBERT → BoW 폴백)
# =========================
class SimpleEmbedder:
    """
    백엔드:
      - sentence-transformers (권장, EMBED_MODEL)
      - 실패 시 BoW 코사인(간이)
    """
    def __init__(self):
        self.name = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
        self.backend = None
        self.uses_sbert = False
        if _HAS_SENT_EMB:
            try:
                self.backend = SentenceTransformer(self.name)
                self.uses_sbert = True
            except Exception:
                self.backend = None

    def _bow_vec(self, text: str) -> Dict[str, float]:
        toks = re.findall(r"[A-Za-z가-힣0-9]+", (text or "").lower())
        d: Dict[str, float] = {}
        for t in toks:
            d[t] = d.get(t, 0.0) + 1.0
        norm = math.sqrt(sum(v*v for v in d.values())) or 1.0
        for k in d:
            d[k] /= norm
        return d

    def _bow_cos(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):  # iterate smaller
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def encode(self, texts: List[str]):
        if not texts:
            return []
        if self.backend is not None:
            return self.backend.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
        return [self._bow_vec(t) for t in texts]

    def cosine(self, a, b) -> float:
        if self.backend is not None:
            # a, b are numpy arrays (unit-normalized)
            return float((a * b).sum())
        return self._bow_cos(a, b)

# =========================
# PDF 인덱스 (PyMuPDF 필요)
# =========================
@dataclass
class EvidenceChunk:
    text: str
    page: int
    sim: float
    source: str

class PDFIndex:
    def __init__(self, embedder: Optional[SimpleEmbedder] = None, chunk_chars: int = 600, overlap: int = 100):
        self.embedder = embedder or SimpleEmbedder()
        self.chunk_chars = int(os.environ.get("PDF_CHUNK_CHARS", chunk_chars))
        self.overlap = int(os.environ.get("PDF_CHUNK_OVERLAP", overlap))
        self.chunks: List[EvidenceChunk]
