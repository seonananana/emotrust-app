# acc_score.py
# B안: PDF 기반 팩트체크 (Claim → Retrieval → NLI) + 증거 없으면 업로드 유도
# - 선택 의존성(있으면 품질↑): pymupdf, pypdf, pytesseract(+tesseract-ocr-kor),
#   sentence-transformers, transformers
# - 의존성 없을 때는 BoW 유사도로 폴백

import os
import re
import io
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
_HAS_PYPDF = False
_HAS_TESS = False
_HAS_SENT_EMB = False
_HAS_NLI = False

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
    # 마침표/물음표/느낌표 + 줄바꿈
    s = re.sub(r"([\.!?])", r"\1\n", text)
    parts = [p.strip() for p in s.splitlines() if p.strip()]
    return parts

# 존댓말 문체까지 포함 (합니다/드립니다/됩니다 등)
_CLAIM_ENDING_RE = re.compile(
    r"(이다|였다|한다|한다\.|합니다|드립니다|됩니다|임\.?|라고 하|로 밝혀)"
)

def _is_claim_like(sent: str) -> bool:
    if not sent or len(sent) < 12:
        return False
    # 숫자/날짜/단위
    if re.search(r"\d{4}년|\d{1,2}월|\d{1,2}일|[\d,]+|%|억원|만명|천명|km|kg|원", sent):
        return True
    if _CLAIM_ENDING_RE.search(sent):
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
# PDF 텍스트 추출 (경로/바이트 + OCR 폴백)
# =========================
def _extract_page_texts_from_pdf(
    *, 
    path: Optional[str] = None, 
    pdf_bytes: Optional[bytes] = None,
    ocr_lang: str = "kor+eng",
    ocr_min_len: int = 20,
    render_scale: float = 2.0,
) -> List[str]:
    """
    1) PyMuPDF 있으면 그대로 사용 (경로 또는 바이트 스트림)
    2) 텍스트가 거의 없으면 해당 페이지를 이미지 렌더 → Tesseract OCR (kor+eng)
    3) PyMuPDF 없으면 pypdf로 텍스트만 추출 (OCR 없음)
    """
    pages_text: List[str] = []

    # 1) PyMuPDF 경로/바이트 오픈
    if _HAS_PYMUPDF:
        try:
            if pdf_bytes is not None:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                if not path:
                    raise ValueError("No path provided")
                doc = fitz.open(path)
            for page in doc:
                text = page.get_text("text") or ""
                # OCR 폴백: 텍스트가 너무 짧으면 이미지 렌더링 후 OCR
                if _HAS_TESS and len(text.strip()) < ocr_min_len:
                    try:
                        mat = fitz.Matrix(render_scale, render_scale)
                        pix = page.get_pixmap(matrix=mat)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        try:
                            ocr_txt = pytesseract.image_to_string(img, lang=ocr_lang)
                        except Exception:
                            ocr_txt = pytesseract.image_to_string(img)
                        text = (text + "\n" + (ocr_txt or "")).strip()
                    except Exception:
                        # OCR 렌더 실패 시 그냥 통과
                        pass
                pages_text.append(text)
            try:
                doc.close()
            except Exception:
                pass
            return pages_text
        except Exception:
            pass

    # 2) pypdf 폴백(텍스트만, OCR 없음)
    if _HAS_PYPDF:
        try:
            if pdf_bytes is not None:
                reader = PdfReader(io.BytesIO(pdf_bytes))
            else:
                if not path:
                    raise ValueError("No path provided")
                reader = PdfReader(path)
            for p in reader.pages:
                text = p.extract_text() or ""
                pages_text.append(text)
            return pages_text
        except Exception:
            pass

    # 3) 전부 실패
    return []

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
            self._vecs = []
            return
        texts = [c.text for c in self.chunks]
        self._vecs = self.embedder.encode(texts)

    def search(self, query: str, k: int = 5) -> List[EvidenceChunk]:
        """질의문에 가장 유사한 증거 청크 k개 반환"""
        if not self.chunks:
            return []
        q_vec = self.embedder.encode([_normalize_spaces(query)])[0]
        if self._vecs is None:
            self.build()

        scored: List[Tuple[float, int]] = []
        for idx, vec in enumerate(self._vecs or []):
            sim = self.embedder.cosine(q_vec, vec)
            scored.append((sim, idx))
        scored.sort(reverse=True, key=lambda x: x[0])

        out: List[EvidenceChunk] = []
        for sim, idx in scored[:k]:
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
    (선택) transformers NLI가 있으면 entailment 확률로 대체 가능 (TODO)
    """
    if not evs:
        return 0.0
    sims = [max(0.0, min(1.0, e.sim)) for e in evs]
    sims = sims[:3] if len(sims) >= 3 else sims  # 상위 3개 평균
    return float(sum(sims) / max(1, len(sims)))

# =========================
# 외부에서 호출하는 단일 엔트리포인트
# =========================
def score_with_pdf(
    clean_text: str, 
    pdf_paths: Optional[List[str]] = None,
    pdf_blobs: Optional[List[Tuple[str, bytes]]] = None,  # (파일명, 바이트)
    topk: int = 5
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
    claims = extract_claims(clean_text, max_claims=3)

    # PDF 준비 (경로/바이트 모두 지원)
    idx = PDFIndex()
    idx.load_pdfs(pdf_paths=pdf_paths, pdf_blobs=pdf_blobs)
    idx.build()

    evidence_map: Dict[str, List[Dict[str, Any]]] = {}
    claim_scores: List[float] = []

    has_pdf_index = bool(idx.chunks)

    for cl in claims:
        ev_chunks: List[EvidenceChunk] = idx.search(cl, k=topk) if has_pdf_index else []
        evidence_map[cl] = [
            {"text": e.text, "page": e.page, "sim": float(e.sim), "source": e.source}
            for e in ev_chunks
        ]
        score = _score_claim_with_evidence(cl, ev_chunks) if has_pdf_index else None
        if score is not None:
            claim_scores.append(score)

    if has_pdf_index and claim_scores:
        S_fact = clamp01(sum(claim_scores) / len(claim_scores))
        need_evidence = False
    else:
        S_fact = None
        need_evidence = True if claims else False

    return {
        "S_fact": S_fact,
        "need_evidence": need_evidence,
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
