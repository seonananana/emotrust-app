# acc_score.py
# B안: PDF 기반 팩트체크 (Claim → Retrieval → NLI) + 증거 없으면 업로드 유도
# - 선택 의존성(있으면 품질↑): pymupdf, sentence-transformers, transformers
# - 의존성 없을 때는 보워드(BoW) 유사도/근사 NLI로 폴백

import os
import re
import math
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any  # ← Any 추가

# =========================
# 내부 유틸
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
# 증거 청크 구조
# =========================
@dataclass
class EvidenceChunk:
    text: str
    page: int
    sim: float
    source: str

# =========================
# PDF 텍스트 청크 유틸
# =========================
def _chunk_text(txt: str, chunk_chars: int, overlap: int) -> List[str]:
    txt = _normalize_spaces(txt or "")
    if not txt:
        return []
    chunks = []
    step = max(1, chunk_chars - overlap)
    for i in range(0, len(txt), step):
        chunk = txt[i:i + chunk_chars]
        if chunk:
            chunks.append(chunk)
    return chunks

# =========================
# PDF 인덱스: 로드/인덱스/검색
# =========================
class PDFIndex:
    def __init__(self, embedder: Optional[SimpleEmbedder] = None, chunk_chars: int = 600, overlap: int = 100):
        self.embedder = embedder or SimpleEmbedder()
        self.chunk_chars = int(os.environ.get("PDF_CHUNK_CHARS", chunk_chars))
        self.overlap = int(os.environ.get("PDF_CHUNK_OVERLAP", overlap))
        self.chunks: List[EvidenceChunk] = []
        self._vecs = None  # SBERT 벡터 or BoW 벡터

    def load_pdfs(self, pdf_paths: List[str]) -> None:
        """PDF 파일들을 읽어 EvidenceChunk 리스트를 구성"""
        if not pdf_paths:
            self.chunks = []
            return

        if not _HAS_PYMUPDF:
            # PyMuPDF 없으면 PDF 로딩 불가
            self.chunks = []
            return

        chunks: List[EvidenceChunk] = []
        for path in pdf_paths:
            if not os.path.exists(path):
                continue
            try:
                doc = fitz.open(path)  # type: ignore[name-defined]
            except Exception:
                continue

            base = os.path.basename(path)
            for pno, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text")
                except Exception:
                    text = ""
                for c in _chunk_text(text, self.chunk_chars, self.overlap):
                    chunks.append(EvidenceChunk(text=c, page=pno, sim=0.0, source=base))
            try:
                doc.close()
            except Exception:
                pass

        self.chunks = chunks

    def build(self) -> None:
        """청크 임베딩 미리 계산"""
        if not self.chunks:
            self._vecs = []
            return
        texts = [c.text for c in self.chunks]
        self._vecs = self.embedder.encode(texts)

    def search(self, query: str, k: int = 5) -> List[EvidenceChunk]:
        """질의문에 가장 유사한 증거 청크 k개 반환"""
        if not self.chunks:
            return []
        q_vec = self.embedder.encode([_normalize_spaces(query)])[0]
        scored: List[Tuple[float, int]] = []
        if self._vecs is None:
            self.build()
        # _vecs 는 SBERT 벡터(넘파이) 또는 BoW dict들의 리스트
        for idx, vec in enumerate(self._vecs or []):
            sim = self.embedder.cosine(q_vec, vec)
            scored.append((sim, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:k]
        out: List[EvidenceChunk] = []
        for sim, idx in top:
            c = self.chunks[idx]
            out.append(EvidenceChunk(text=c.text, page=c.page, sim=float(sim), source=c.source))
        return out

# =========================
# 간이 Claim → Evidence → 점수화
# =========================
def _score_claim_with_evidence(claim: str, evs: List[EvidenceChunk]) -> float:
    """
    매우 간단한 스코어러:
      - 상위 evidence 유사도들의 평균을 0~1로 클램프하여 사용
    (선택) transformers NLI가 있으면 entailment 확률로 대체 가능
    """
    if not evs:
        return 0.0
    sims = [max(0.0, min(1.0, e.sim)) for e in evs]
    sims = sims[:3] if len(sims) >= 3 else sims  # 상위 3개 평균
    return float(sum(sims) / max(1, len(sims)))

# =========================
# 외부에서 호출하는 단일 엔트리포인트
# =========================
def score_with_pdf(clean_text: str, pdf_paths: Optional[List[str]] = None) -> Dict[str, Any]:  # ← Any로 수정
    """
    clean_text에서 주장문을 추출 → PDF에서 증거 검색 → 정확성 점수 산출
    반환:
      {
        "S_fact": 0~1 또는 None,
        "need_evidence": bool,
        "claims": [str, ...],
        "evidence": {claim: [{"text":..., "page":..., "sim":..., "source":...}, ...], ...}
      }
    """
    claims = extract_claims(clean_text, max_claims=3)

    # PDF 준비
    idx = PDFIndex()
    if pdf_paths:
        idx.load_pdfs(pdf_paths)
        idx.build()

    evidence_map: Dict[str, List[Dict[str, Any]]] = {}
    claim_scores: List[float] = []

    # PDF가 없거나 로딩 실패한 경우
    has_pdf_index = bool(pdf_paths) and bool(idx.chunks)

    for cl in claims:
        ev_chunks: List[EvidenceChunk] = idx.search(cl, k=5) if has_pdf_index else []
        evidence_map[cl] = [
            {"text": e.text, "page": e.page, "sim": float(e.sim), "source": e.source}
            for e in ev_chunks
        ]
        score = _score_claim_with_evidence(cl, ev_chunks) if has_pdf_index else None
        if score is not None:
            claim_scores.append(score)

    # 최종 S_fact
    if has_pdf_index and claim_scores:
        S_fact = clamp01(sum(claim_scores) / len(claim_scores))
        need_evidence = False
    else:
        # 증거 없으면 None 반환하고 업로드/추가 요청 유도
        S_fact = None
        need_evidence = True if claims else False

    return {
        "S_fact": S_fact,
        "need_evidence": need_evidence,
        "claims": claims,
        "evidence": evidence_map
    }
