from __future__ import annotations

import os
import io
import re
import math
import unicodedata
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

# 선택 의존성
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except:
    fitz = None
    _HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except:
    PdfReader = None
    _HAS_PYPDF = False

try:
    import pytesseract
    from PIL import Image
    _HAS_TESS = True
except:
    pytesseract = None
    Image = None
    _HAS_TESS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except:
    SentenceTransformer = None
    _HAS_SBERT = False

# 텍스트 유틸
_WS_RE = re.compile(r'\s+')
_SENT_SPLIT_RE = re.compile(r'(?<=[\.!?])\s+|[\n\r]+|(?<=[가-힣\w\)])\s*[\-–·]\s+')

def _normalize_spaces(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", s)).strip()

def _split_sentences(s: str) -> List[str]:
    return [p.strip() for p in _SENT_SPLIT_RE.split(_normalize_spaces(s)) if len(p.strip()) > 1]

def _chunk_text(s: str, chunk_chars: int = 600, overlap: int = 100) -> List[str]:
    s = _normalize_spaces(s)
    step = max(1, chunk_chars - overlap)
    return [s[i:i+chunk_chars] for i in range(0, len(s), step)]

# OCR 포함 PDF 텍스트 추출

def _extract_page_texts_from_pdf(path: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> List[str]:
    texts = []
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf") if pdf_bytes else fitz.open(path)
            for page in doc:
                t = page.get_text("text").strip()
                if len(t) < 30 and _HAS_TESS:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    t_ocr = pytesseract.image_to_string(img, lang="kor+eng")
                    t = t_ocr if len(t_ocr) > len(t) else t
                texts.append(_normalize_spaces(t))
            doc.close()
            return texts
        except Exception as e:
            print(f"OCR 추출 실패: {e}")

    if _HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes)) if pdf_bytes else PdfReader(path)
            for page in reader.pages:
                texts.append(_normalize_spaces(page.extract_text() or ""))
            return texts
        except Exception as e:
            print(f"pypdf 실패: {e}")

    return texts

# 간단 임베딩 (SBERT or BoW)
class SimpleEmbedder:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.uses_sbert = _HAS_SBERT
        self.backend = SentenceTransformer(model_name) if _HAS_SBERT else None

    def encode(self, texts):
        if self.backend:
            return self.backend.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
        return [self._bow_vec(t) for t in texts]

    def cosine(self, a, b):
        if self.backend:
            return float((a * b).sum())
        return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a if k in b)

    def _bow_vec(self, s):
        toks = re.findall(r"[가-힣a-zA-Z0-9]+", s.lower())
        vec = {t: toks.count(t) for t in set(toks)}
        norm = math.sqrt(sum(v*v for v in vec.values()))
        return {k: v/norm for k, v in vec.items()} if norm else vec

@dataclass
class EvidenceChunk:
    text: str
    page: int
    sim: float
    source: str

class PDFIndex:
    def __init__(self, embedder=None, chunk_chars=600, overlap=100):
        self.embedder = embedder or SimpleEmbedder()
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.chunks = []
        self.embeds = []

    def load(self, pdf_blobs: List[Tuple[str, bytes]]):
        for name, blob in pdf_blobs:
            texts = _extract_page_texts_from_pdf(pdf_bytes=blob)
            for page_num, text in enumerate(texts, 1):
                for chunk in _chunk_text(text, self.chunk_chars, self.overlap):
                    self.chunks.append(EvidenceChunk(chunk, page_num, 0.0, name))
        self.embeds = self.embedder.encode([c.text for c in self.chunks])

    def search(self, query, k=5):
        q_embed = self.embedder.encode([query])[0]
        scored = [(self.embedder.cosine(q_embed, e), c) for e, c in zip(self.embeds, self.chunks)]
        return sorted(scored, reverse=True)[:k]

def extract_claims(text, max_claims=3):
    sents = [s for s in _split_sentences(text) if len(s) > 7]
    return sents[:max(1, max_claims)] if sents else [text.strip()]

def score_with_pdf(clean_text, pdf_blobs):
    idx = PDFIndex()
    idx.load(pdf_blobs)

    if not idx.chunks:
        return {"S_fact": None, "need_evidence": True, "claims": [], "evidence": {}, "meta": {}}

    claims = extract_claims(clean_text)
    per_scores, evid_map = [], {}
    for c in claims:
        top = idx.search(c)
        best = top[0][0] if top else 0.0
        per_scores.append(best if best > 0 else 0.01)
        evid_map[c] = [{"text": ec.text, "sim": s, "page": ec.page, "source": ec.source} for s, ec in top]

    s_fact = sum(per_scores)/len(per_scores)
    return {
        "S_fact": round(s_fact, 4),
        "need_evidence": s_fact < 0.3,
        "claims": claims,
        "evidence": evid_map,
        "meta": {"chunks": len(idx.chunks)}
    }
