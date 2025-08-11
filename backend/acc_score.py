# acc_score.py
# B안: PDF 기반 팩트체크 (Claim → Retrieval → NLI) + '증거 없으면 업로드 유도' 플래그 포함
# - 선택 의존성(있으면 품질↑): PyMuPDF, sentence-transformers, transformers
# - 다른 공용 유틸/PII/전처리/사전 로직은 포함하지 않음(파이프라인 외부에서 처리 가정)

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
    """간이 문장 분리 (정교함 필요하면 kss 등으로 교체)"""
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
      - sentence-transformers (권장, EMBED_MODEL 환경변수)
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
            import numpy as np
            a = np.asarray(a); b = np.asarray(b)
            return float((a * b).sum())  # 이미 정규화됨
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
        self.chunks: List[EvidenceChunk] = []
        self._vecs = None

    def _chunk_page(self, text: str, page: int, source: str) -> List[EvidenceChunk]:
        text = _normalize_spaces(text)
        if not text:
            return []
        L, O = self.chunk_chars, self.overlap
        out: List[EvidenceChunk] = []
        i = 0
        while i < len(text):
            j = min(len(text), i + L)
            out.append(EvidenceChunk(text=text[i:j], page=page, sim=0.0, source=source))
            if j == len(text): break
            i = max(0, j - O)
        return out

    def add_pdf(self, pdf_path: str, source: Optional[str] = None):
        if not _HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF(fitz) 미설치: pip install pymupdf")
        source = source or os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        try:
            for pno in range(len(doc)):
                page = doc.load_page(pno)
                txt = page.get_text("text")
                for ch in self._chunk_page(txt, pno + 1, source):
                    self.chunks.append(ch)
        finally:
            doc.close()
        texts = [c.text for c in self.chunks]
        self._vecs = self.embedder.encode(texts)

    def search(self, query: str, topk: int = 5) -> List[EvidenceChunk]:
        if not self.chunks:
            return []
        qv = self.embedder.encode([query])[0]
        sims: List[Tuple[float, int]] = []
        for i, vec in enumerate(self._vecs):
            sims.append((self.embedder.cosine(qv, vec), i))
        sims.sort(reverse=True, key=lambda x: x[0])
        hits: List[EvidenceChunk] = []
        for s, idx in sims[:max(1, min(topk, len(sims)))]:
            c = self.chunks[idx]
            hits.append(EvidenceChunk(text=c.text, page=c.page, sim=float(s), source=c.source))
        return hits

# =========================
# NLI (제로샷) / 폴백(유사도 근사)
# =========================
class NLIScorer:
    """XNLI 제로샷(있으면) / 유사도 근사(없으면)"""
    def __init__(self):
        self.ok = False
        self.tok = None
        self.model = None
        self.embedder = SimpleEmbedder()
        model_name = os.environ.get("NLI_MODEL", "joeddav/xlm-roberta-large-xnli")
        if _HAS_NLI:
            try:
                self.tok = _NLI_Tok.from_pretrained(model_name)
                self.model = _NLI_Model.from_pretrained(model_name).eval()
                # id2label에서 인덱스 매핑 확보
                id2label = getattr(self.model.config, "id2label", None)
                if isinstance(id2label, dict):
                    self.idx_ent = int([k for k, v in id2label.items() if str(v).lower().startswith("entail")][0])
                    self.idx_neu = int([k for k, v in id2label.items() if str(v).lower().startswith("neutral")][0])
                    self.idx_con = int([k for k, v in id2label.items() if str(v).lower().startswith("contra")][0])
                else:
                    # 일반적 순서 가정: [entail, neutral, contradict]
                    self.idx_ent, self.idx_neu, self.idx_con = 0, 1, 2
                self.ok = True
            except Exception:
                self.ok = False

    def probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        if self.ok and self.tok and self.model:
            import torch
            with torch.inference_mode():
                batch = self.tok(premise, hypothesis, truncation=True, return_tensors="pt", max_length=384)
                logits = self.model(**batch).logits.squeeze(0)
                probs = logits.softmax(-1).tolist()
                return {
                    "entail": float(probs[self.idx_ent]),
                    "neutral": float(probs[self.idx_neu]),
                    "contrad": float(probs[self.idx_con]),
                }
        # 폴백: 임베딩 유사도 근사
        a, b = self.embedder.encode([premise, hypothesis])
        sim = self.embedder.cosine(a, b)  # [-1,1] 유사도 가정
        sim = max(-1.0, min(1.0, sim))
        entail = (sim + 1.0) / 2.0
        contrad = (1.0 - sim) / 2.0
        neutral = max(0.0, 1.0 - entail - contrad)
        return {"entail": entail, "neutral": neutral, "contrad": contrad}

# =========================
# 집계
# =========================
def aggregate_fact_score(claims: List[str], evid_map: Dict[str, List[EvidenceChunk]], nli: NLIScorer) -> Tuple[float, Dict[str, float]]:
    claim_scores: Dict[str, float] = {}
    per = []
    for c in claims:
        chunks = evid_map.get(c, [])
        if not chunks:
            claim_scores[c] = 0.5  # 근거 없음 → 중립
            per.append(0.5)
            continue
        num = 0.0; den = 0.0
        for ch in chunks:
            pr = nli.probs(premise=ch.text, hypothesis=c)
            w = max(0.0, float(ch.sim))  # 검색 유사도 가중
            num += w * (pr["entail"] - pr["contrad"])
            den += w
        s = (num / den) if den > 0 else 0.0      # [-1,1]
        s01 = clamp01((s + 1.0) / 2.0)           # [0,1]
        claim_scores[c] = s01
        per.append(s01)
    S_fact = float(sum(per) / len(per)) if per else 0.0
    return clamp01(S_fact), claim_scores

# =========================
# 공개 엔트리포인트
# =========================
def score_with_pdf(clean_text: str,
                   pdf_paths: Optional[List[str]] = None,
                   max_claims: int = 3,
                   topk: int = 5) -> Dict:
    """
    B안 메인:
      입력: 전처리/PII 완료된 텍스트(clean_text), PDF 경로 리스트(없을 수 있음)
      출력:
        {
          "claims": [...],
          "evidence": {claim: [{"page":..,"sim":..,"source":..,"snippet":..}, ...]},
          "per_claim": {claim: score01},
          "S_fact": 0~1 or None,
          "need_evidence": bool
        }
    """
    clean_text = _normalize_spaces(clean_text or "")
    claims = extract_claims(clean_text, max_claims=max_claims)

    if not claims:
        return {
            "claims": [],
            "evidence": {},
            "per_claim": {},
            "S_fact": None,
            "need_evidence": bool(pdf_paths),  # PDF가 있어도 주장 없으면 스킵
            "note": "no_claim"
        }

    if not pdf_paths or len(pdf_paths) == 0:
        # 정밀 검증 불가 → 증거 업로드 유도
        return {
            "claims": claims,
            "evidence": {},
            "per_claim": {},
            "S_fact": None,
            "need_evidence": True,
            "note": "pdf_required_for_fact_check"
        }

    # PDF 인덱싱
    idx = PDFIndex()
    for p in pdf_paths:
        idx.add_pdf(p, source=os.path.basename(p))

    # 검색
    evid_map: Dict[str, List[EvidenceChunk]] = {}
    for c in claims:
        evid_map[c] = idx.search(c, topk=topk)

    # NLI/폴백
    nli = NLIScorer()
    S_fact, per_claim = aggregate_fact_score(claims, evid_map, nli)

    # 스니펫 정리
    snippets = {
        c: [
            {
                "page": e.page,
                "sim": round(e.sim, 3),
                "source": e.source,
                "snippet": e.text[:220]
            } for e in evid_map.get(c, [])
        ]
        for c in claims
    }

    return {
        "claims": claims,
        "evidence": snippets,
        "per_claim": per_claim,
        "S_fact": S_fact,
        "need_evidence": False
    }
