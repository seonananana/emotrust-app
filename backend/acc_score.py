
# acc_score.py
# PDF 기반 정확성(팩트) 스코어러
# - 주장 추출 → PDF 인덱싱 → 검색 → 유사도 평균으로 S_fact 산출
# - PyMuPDF(있으면 우선), pypdf(폴백), pytesseract(텍스트 부족 시 OCR), sentence-transformers(임베딩)가 있으면 사용
# - 없으면 BoW 코사인으로 폴백
from __future__ import annotations

import os
import io
import re
import math
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

# =========================
# 내부 유틸
# =========================
def clamp01(x) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

_WS_RE = re.compile(r'\s+')

def _normalize_spaces(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return _WS_RE.sub(" ", s).strip()

_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+|[\n\r]+|(?<=[가-힣\w\)])\s*[·\-–—]\s+')

def _split_sentences(s: str) -> List[str]:
    s = _normalize_spaces(s)
    if not s:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(s) if p and len(p.strip()) > 1]
    return parts

def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        k = x.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def _tokenize_kor_en(s: str) -> List[str]:
    # 간단 토크나이저: 한글/영문/숫자 기준
    s = _normalize_spaces(s.lower())
    # 한글/영문/숫자 외에는 공백으로
    s = re.sub(r"[^0-9a-z가-힣]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks

def _chunk_text(s: str, chunk_chars: int = 600, overlap: int = 100) -> List[str]:
    """문자열을 길이 chunk_chars로 겹치게(overlap) 잘라 반환"""
    s = _normalize_spaces(s)
    if not s:
        return []
    chunk_chars = max(1, int(chunk_chars))
    overlap = max(0, int(overlap))
    step = max(1, chunk_chars - overlap)
    return [s[i:i + chunk_chars] for i in range(0, len(s), step)]

# =========================
# 선택 의존성
# =========================
_HAS_PYMUPDF = False
_HAS_PYPDF = False
_HAS_TESS = False
_HAS_SBERT = False

fitz = None
PdfReader = None
SentenceTransformer = None
pytesseract = None
Image = None

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    pass

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except Exception:
    pass

try:
    import pytesseract
    from PIL import Image
    _HAS_TESS = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    pass

# =========================
# 임베더
# =========================
class SimpleEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.uses_sbert = False
        self.backend = None
        if _HAS_SBERT:
            try:
                self.backend = SentenceTransformer(model_name)
                self.uses_sbert = True
            except Exception:
                self.backend = None
                self.uses_sbert = False

    # BoW 전용 내부 벡터
    def _bow_vec(self, s: str) -> Dict[str, float]:
        toks = _tokenize_kor_en(s)
        if not toks:
            return {}
        # tf / sqrt(len)
        d: Dict[str, float] = {}
        for t in toks:
            d[t] = d.get(t, 0.0) + 1.0
        norm = math.sqrt(sum(v*v for v in d.values()))
        if norm == 0:
            return d
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
# PDF 텍스트 추출
# =========================
def _extract_page_texts_from_pdf(path: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> List[str]:
    """
    각 페이지의 텍스트를 리스트로 반환.
    우선순위: PyMuPDF → pypdf. PyMuPDF 사용 시 텍스트가 매우 적으면 OCR 시도.
    """
    texts: List[str] = []

    # PyMuPDF 경로
    if _HAS_PYMUPDF:
        try:
            if pdf_bytes is not None:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                assert path is not None, "path or pdf_bytes required"
                doc = fitz.open(path)
            for page in doc:
                t = page.get_text("text") or ""
                t = _normalize_spaces(t)
                # 텍스트가 지나치게 적으면 OCR 시도
                if len(t) < 30 and _HAS_TESS:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_bytes))
                        lang = "kor+eng"
                        t_ocr = pytesseract.image_to_string(img, lang=lang)
                        t2 = _normalize_spaces(t_ocr)
                        if len(t2) > len(t):
                            t = t2
                    except Exception:
                        pass
                texts.append(t)
            doc.close()
            return texts
        except Exception:
            # fall through to pypdf
            pass

    # pypdf 폴백
    if _HAS_PYPDF:
        try:
            if pdf_bytes is not None:
                reader = PdfReader(io.BytesIO(pdf_bytes))
            else:
                assert path is not None, "path or pdf_bytes required"
                reader = PdfReader(path)
            for p in reader.pages:
                t = p.extract_text() or ""
                texts.append(_normalize_spaces(t))
            return texts
        except Exception:
            pass

    # 둘 다 실패하면 빈 리스트
    return texts

# =========================
# PDF 인덱스
# =========================
class PDFIndex:
    def __init__(self, embedder: Optional[SimpleEmbedder] = None, chunk_chars: int = 600, overlap: int = 100):
        self.embedder = embedder or SimpleEmbedder()
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.chunks: List[EvidenceChunk] = []
        self._chunk_embeds = None

    def load_pdfs(
        self, 
        pdf_paths: Optional[List[str]] = None, 
        pdf_blobs: Optional[List[Tuple[str, bytes]]] = None  # (파일명, 바이트)
    ) -> None:
        """경로 및 메모리 바이트 입력을 모두 지원"""
        chunks: List[EvidenceChunk] = []

        # 1) 경로 기반
        for path in (pdf_paths or []):
            if not path or not os.path.exists(path):
                continue
            texts = _extract_page_texts_from_pdf(path=path)
            base = os.path.basename(path)
            for pno, text in enumerate(texts, start=1):
                for c in _chunk_text(text, self.chunk_chars, self.overlap):
                    if c:
                        chunks.append(EvidenceChunk(text=c, page=pno, sim=0.0, source=base))

        # 2) 메모리(바이트) 기반
        for name, blob in (pdf_blobs or []):
            texts = _extract_page_texts_from_pdf(pdf_bytes=blob)
            base = name or "uploaded.pdf"
            for pno, text in enumerate(texts, start=1):
                for c in _chunk_text(text, self.chunk_chars, self.overlap):
                    if c:
                        chunks.append(EvidenceChunk(text=c, page=pno, sim=0.0, source=base))

        self.chunks = chunks

    def build(self) -> None:
        """청크 임베딩 미리 계산"""
        if not self.chunks:
            self._chunk_embeds = []
            return
        texts = [c.text for c in self.chunks]
        self._chunk_embeds = self.embedder.encode(texts)

    def search(self, query: str, k: int = 5) -> List[EvidenceChunk]:
        """쿼리와 가장 유사한 청크 상위 k개 반환 (심플 코사인)"""
        if not self.chunks:
            return []
        if self._chunk_embeds is None:
            self.build()
        q_embed = self.embedder.encode([query])[0]
        scored: List[Tuple[float, EvidenceChunk]] = []
        for emb, ch in zip(self._chunk_embeds, self.chunks):
            sim = self.embedder.cosine(q_embed, emb)
            scored.append((sim, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, ch in scored[:max(1, k)]:
            out.append(EvidenceChunk(text=ch.text, page=ch.page, sim=float(sim), source=ch.source))
        return out

# =========================
# 주장 추출
# =========================
def extract_claims(clean_text: str, max_claims: int = 3) -> List[str]:
    """
    입력 텍스트에서 주요 주장(문장) 추출 (간이 버전)
    - 문장 단위로 분할 → 너무 짧은 문장 제외 → 앞쪽에서 max_claims개
    """
    sents = _split_sentences(clean_text or "")
    sents = [s for s in sents if len(s) >= 8]
    sents = _dedupe_keep_order(sents)
    if not sents and clean_text:
        sents = [clean_text.strip()]
    return sents[:max(1, max_claims)]

# =========================
# 스코어링
# =========================
def _score_claim_with_evidence(claim: str, idx: PDFIndex, topk: int = 5) -> Tuple[float, List[EvidenceChunk]]:
    """
    한 개 주장에 대해 PDFIndex에서 topk 증거를 찾고 최고 유사도를 반환
    """
    evids = idx.search(claim, k=topk) if idx else []
    best = max((e.sim for e in evids), default=0.0)
    return float(best), evids

def score_with_pdf(
    clean_text: str,
    pdf_paths: Optional[List[str]] = None,
    pdf_blobs: Optional[List[Tuple[str, bytes]]] = None,
    topk: int = 5,
    chunk_chars: int = 600,
    overlap: int = 100,
) -> Dict[str, Any]:
    """
    clean_text에서 주장문을 추출 → PDF에서 증거 검색 → 정확성 점수 산출
    반환:
      {
        "S_fact": 0~1 또는 None,
        "need_evidence": bool,
        "claims": [str, ...],
        "evidence": {claim: [{"text":..., "page":..., "sim":..., "source":...}, ...], ...},
        "meta": {...}
      }
    """
    idx = PDFIndex(chunk_chars=chunk_chars, overlap=overlap)
    idx.load_pdfs(pdf_paths=pdf_paths, pdf_blobs=pdf_blobs)

    claims = extract_claims(clean_text, max_claims=3)

    if not idx.chunks:
        # 증거가 하나도 없으면 S_fact는 None, 증거 필요 플래그 ON
        return {
            "S_fact": None,
            "need_evidence": True if claims else False,
            "claims": claims,
            "evidence": {},
            "meta": {
                "chunks": 0,
                "uses_sbert": idx.embedder.uses_sbert,
                "has_pymupdf": _HAS_PYMUPDF,
                "has_pypdf": _HAS_PYPDF,
                "has_tesseract": _HAS_TESS,
            }
        }

    per_claim_scores: List[float] = []
    evidence_map: Dict[str, List[Dict[str, Any]]] = {}

    for claim in claims:
        best, evs = _score_claim_with_evidence(claim, idx, topk=topk)
        per_claim_scores.append(best)
        evidence_map[claim] = [
            {"text": e.text, "page": e.page, "sim": float(e.sim), "source": e.source}
            for e in evs
        ]

    S_fact = None
    if per_claim_scores:
        S_fact = float(sum(per_claim_scores) / len(per_claim_scores))
        S_fact = clamp01(S_fact)

    need_evidence = (S_fact is None) or (S_fact < 0.3 and any(len(v) == 0 for v in evidence_map.values()))

    return {
        "S_fact": S_fact,
        "need_evidence": bool(need_evidence),
        "claims": claims,
        "evidence": evidence_map,
        "meta": {
            "chunks": len(idx.chunks),
            "uses_sbert": idx.embedder.uses_sbert,
            "has_pymupdf": _HAS_PYMUPDF,
            "has_pypdf": _HAS_PYPDF,
            "has_tesseract": _HAS_TESS,
        }
    }
