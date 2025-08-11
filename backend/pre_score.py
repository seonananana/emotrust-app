# pre_score.py
# -*- coding: utf-8 -*-

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

# 숫자/기호 제외(영문/한글만)
_word_re = re.compile(r"[A-Za-z가-힣]+", re.UNICODE)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class LexiconScorer:
    """
    NRC-Korean 유사 포맷 CSV에서 (단어 → 단일 점수) 사전을 구성하고
    입력 텍스트의 '진정성(0~1)'을 계산한다.

    기대 CSV 헤더(기본):
      - Korean Word
      - Emotion
      - Emotion-Intensity-Score

    한 단어에 여러 감정 행이 존재할 수 있으므로
    단어별 점수는 모든 감정 강도의 '평균'으로 집계한다.
    """

    def __init__(
        self,
        path: str,
        word_col: Optional[str] = None,
        score_col: Optional[str] = None,
    ):
        self.vocab: Dict[str, float] = {}
        self.min_v: float = 0.0
        self.max_v: float = 1.0
        self._load(path, word_col, score_col)

    def _load(self, path: str, word_col: Optional[str], score_col: Optional[str]) -> None:
        # 컬럼 기본값: 사용자가 제공한 헤더 고정
        wcol = (word_col or os.environ.get("EMO_LEX_WORD_COL") or "Korean Word")
        scol = (score_col or os.environ.get("EMO_LEX_SCORE_COL") or "Emotion-Intensity-Score")

        buckets = defaultdict(list)  # word -> [scores...]

        with open(path, "r", encoding="utf-8") as f:
            # 구분자 자동 감지 (실패 시 기본 excel)
            try:
                sample = f.read(2048)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            except Exception:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                if not row:
                    continue
                w = str(row.get(wcol, "")).strip().lower()
                if not w:
                    continue
                val = row.get(scol, "")
                try:
                    s = float(val)
                except Exception:
                    continue
                if math.isfinite(s):
                    buckets[w].append(s)

        # 단어별 평균 강도
        self.vocab = {w: (sum(vals) / len(vals)) for w, vals in buckets.items() if vals}

        # 정규화 범위 기록(사전이 이미 0~1 스케일이면 그대로 유지)
        if self.vocab:
            self.min_v = min(self.vocab.values())
            self.max_v = max(self.vocab.values())

    def _norm(self, x: float) -> float:
        # 사전 점수가 [0,1] 범위를 벗어나면 min–max 정규화
        if self.min_v < 0.0 or self.max_v > 1.0:
            if self.max_v == self.min_v:
                return 0.0
            x = (x - self.min_v) / (self.max_v - self.min_v)
        return clamp01(x)

    def sincerity(self, text: str, mode: str = "all", alpha: float = 2.0) -> Tuple[float, int, int, float]:
        """
        진정성 점수 계산
        returns: (score, matched_count, total_tokens, coverage)

        mode='all'     -> 분모 N = 전체 단어 수 + alpha  (보수적)
        mode='matched' -> 분모 N = 사전 일치 단어 수 + alpha
        alpha          -> 라플라스 스무딩 (짧은 글 과대평가 방지)
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


# ===== 전역 로더 & 헬퍼 =====
_lex: Optional[LexiconScorer] = None


def _default_csv_path() -> Path:
    """
    기본 CSV 경로:
      - 환경변수 EMO_LEXICON_PATH 우선
      - 없으면 현재 파일 기준 'backend/data/nrc_words.csv'
    """
    env_path = os.environ.get("EMO_LEXICON_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # pre_score.py 가 backend/ 내에 위치한다고 가정
    here = Path(__file__).resolve().parent
    return (here / "data" / "nrc_words.csv").resolve()  # backend/data/nrc_words.csv


def get_lexicon() -> LexiconScorer:
    global _lex
    if _lex is None:
        csv_path = _default_csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(f"EMO_LEXICON_PATH not found: {csv_path}")
        _lex = LexiconScorer(
            str(csv_path),
            word_col=os.environ.get("EMO_LEX_WORD_COL", "Korean Word"),
            score_col=os.environ.get("EMO_LEX_SCORE_COL", "Emotion-Intensity-Score"),
        )
    return _lex


def sincerity_score(
    text: str,
    mode: str = "all",
    alpha: float = 2.0,
) -> Tuple[float, int, int, float]:
    """
    모듈 외부에서 바로 호출할 수 있는 헬퍼.
    analyzer.py 등에서 import 하여 사용.

    Example:
        from pre_score import sincerity_score
        s, matched, total, cov = sincerity_score(text)
    """
    return get_lexicon().sincerity(text, mode=mode, alpha=alpha)


__all__ = [
    "LexiconScorer",
    "get_lexicon",
    "sincerity_score",
]

