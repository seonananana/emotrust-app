# pre_score.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
import os
import re
import unicodedata as ucd
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

# ===== 설정 =====
# 단어 토큰 정규식 (영문/한글)
_WORD_RE = re.compile(r"[A-Za-z가-힣]+", re.UNICODE)

# 컬럼명 후보(대소문자 무시)
_WORD_ALIASES = {"korean word", "word", "token", "term", "단어"}
_SCORE_ALIASES = {"emotion-intensity-score", "score", "weight", "value", "점수"}

# 환경변수 키
ENV_PATH = "EMO_LEXICON_PATH"
ENV_WORD_COL = "EMO_LEX_WORD_COL"
ENV_SCORE_COL = "EMO_LEX_SCORE_COL"
ENV_DEBUG = "EMO_LEX_DEBUG"


def _dbg_on() -> bool:
    v = os.environ.get(ENV_DEBUG, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _casefold(s: Optional[str]) -> str:
    return (s or "").strip().casefold()


def _pick_col(fieldnames: Iterable[str], user_pref: Optional[str], aliases: set[str]) -> Optional[str]:
    """
    DictReader fieldnames 중에서 사용자가 지정한 컬럼(user_pref) 우선,
    없으면 aliases 집합 중 첫 매칭을 반환.
    """
    if not fieldnames:
        return None
    fields = list(fieldnames)
    # 1) 사용자 지정 우선
    if user_pref:
        for f in fields:
            if _casefold(f) == _casefold(user_pref):
                return f
    # 2) 별칭 매칭
    low_map = {_casefold(f): f for f in fields}
    for a in aliases:
        if a in low_map:
            return low_map[a]
    return None


def _sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample)
    except Exception:
        # 콤마/탭 중 하나로 재시도
        class _ExcelTab(csv.Dialect):
            delimiter = "\t"
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\r\n"
            quoting = csv.QUOTE_MINIMAL
        # 간단한 힌트로 판단
        if "\t" in sample and sample.count("\t") >= sample.count(","):
            return _ExcelTab
        return csv.excel


def _candidate_paths() -> list[Path]:
    """
    CSV 자동 탐색 경로 우선순위:
      1) ENV: EMO_LEXICON_PATH
      2) pre_score.py 기준 ./data/nrc_words.csv
      3) 현재 작업경로 ./data/nrc_words.csv
      4) pre_score.py와 같은 폴더의 ./nrc_words.csv
      5) 현재 작업경로 ./nrc_words.csv
    """
    cands: list[Path] = []
    env_path = os.environ.get(ENV_PATH)
    if env_path:
        cands.append(Path(env_path).expanduser())

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    cands += [
        here / "data" / "nrc_words.csv",
        cwd / "data" / "nrc_words.csv",
        here / "nrc_words.csv",
        cwd / "nrc_words.csv",
    ]

    # 중복 제거 & 정규화
    seen, out = set(), []
    for p in cands:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


@dataclass
class LexiconScorer:
    def __init__(
        self,
        path: str,
        word_col: Optional[str] = None,
        score_col: Optional[str] = None,
    ):
        self.vocab: Dict[str, float] = {}
        self.min_v: float = 0.0
        self.max_v: float = 1.0

        # ⬇️ 추가: 인스턴스 속성으로 보관
        self.path = os.path.abspath(path)
        self.word_col = word_col or os.environ.get("EMO_LEX_WORD_COL") or "Korean Word"
        self.score_col = score_col or os.environ.get("EMO_LEX_SCORE_COL") or "Emotion-Intensity-Score"

        # 기존 로딩
        self._load(self.path, self.word_col, self.score_col)


    # ---------- 로딩 ----------
    @classmethod
    def from_csv(
        cls,
        path: str,
        word_col: Optional[str] = None,
        score_col: Optional[str] = None,
        debug: bool = False,
    ) -> "LexiconScorer":
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Lexicon file not found: {p}")

        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = _sniff_dialect(sample)
            reader = csv.DictReader(f, dialect=dialect)

            if not reader.fieldnames:
                raise ValueError(f"CSV header not found: {p}")

            wcol = _pick_col(reader.fieldnames, word_col or os.environ.get(ENV_WORD_COL), _WORD_ALIASES)
            scol = _pick_col(reader.fieldnames, score_col or os.environ.get(ENV_SCORE_COL), _SCORE_ALIASES)
            if not wcol or not scol:
                raise ValueError(
                    f"Column detection failed. Got header={reader.fieldnames}\n"
                    f"Need word_col in {_WORD_ALIASES} and score_col in {_SCORE_ALIASES} "
                    f"(or set {ENV_WORD_COL}/{ENV_SCORE_COL})."
                )

            buckets: dict[str, list[float]] = defaultdict(list)
            row_cnt = 0
            for row in reader:
                row_cnt += 1
                w_raw = (row.get(wcol) or "").strip()
                if not w_raw:
                    continue
                # NFKC → 소문자
                w = ucd.normalize("NFKC", w_raw).casefold()
                # 점수
                try:
                    s = float(str(row.get(scol, "")).strip())
                except Exception:
                    continue
                if math.isfinite(s):
                    buckets[w].append(s)

        vocab = {w: (sum(vals) / len(vals)) for w, vals in buckets.items() if vals}
        if not vocab:
            raise ValueError(f"No valid rows parsed from CSV: {p} (rows={row_cnt})")

        min_v = min(vocab.values())
        max_v = max(vocab.values())
        if debug:
            print(f"[lexicon] path={p}")
            print(f"[lexicon] rows={row_cnt}, words={len(vocab)}, min_v={min_v:.4f}, max_v={max_v:.4f}")
            print(f"[lexicon] columns -> word_col='{wcol}', score_col='{scol}'")

        return cls(vocab=vocab, path=str(p), word_col=wcol, score_col=scol, min_v=min_v, max_v=max_v)

    # ---------- 전처리 ----------
    @staticmethod
    def _norm_token(t: str) -> str:
        # NFKC 정규화 + 소문자
        return ucd.normalize("NFKC", t).casefold()

    def _norm_score(self, x: float) -> float:
        # [0,1] 범위를 벗어나면 min–max 정규화
        if self.min_v < 0.0 or self.max_v > 1.0:
            if self.max_v == self.min_v:
                return 0.0
            x = (x - self.min_v) / (self.max_v - self.min_v)
        return clamp01(x)

    def _tokenize(self, text: str) -> list[str]:
        return [_WORD_RE.findall,][0](text)  # keep regex object out of hot path

    # ---------- 스코어링 ----------
    def sincerity(
        self,
        text: str,
        mode: str = "all",
        alpha: float = 2.0,
    ) -> Tuple[float, int, int, float]:
        """
        returns: (score, matched_count, total_tokens, coverage)

        mode='all'     -> 분모 N = 전체 단어 수 + alpha  (보수적)
        mode='matched' -> 분모 N = 사전 일치 단어 수 + alpha
        alpha          -> 라플라스 스무딩 (짧은 글 과대평가 방지)
        """
        if not text:
            return 0.0, 0, 0, 0.0

        raw_toks = self._tokenize(text)
        if not raw_toks:
            return 0.0, 0, 0, 0.0

        toks = [self._norm_token(t) for t in raw_toks]
        total = len(toks)

        matched_scores = []
        for t in toks:
            s = self.vocab.get(t)
            if s is not None:
                matched_scores.append(self._norm_score(s))

        matched = len(matched_scores)
        denom = (max(1, matched) if mode == "matched" else total) + alpha
        score = (sum(matched_scores) / denom) if denom > 0 else 0.0
        cov = matched / max(1, total)
        return clamp01(score), matched, total, cov


# ===== 전역 싱글톤 로더 =====
_lex: Optional[LexiconScorer] = None


def _resolve_csv_path() -> Path:
    # 1) 후보 경로 순회
    for p in _candidate_paths():
        if p.exists():
            return p
    # 2) 실패 시 후보 리스트를 메시지에 포함
    tried = "\n  - " + "\n  - ".join(str(p) for p in _candidate_paths())
    raise FileNotFoundError(f"Lexicon CSV not found. Tried:{tried}")


def get_lexicon() -> LexiconScorer:
    """모듈 전역 캐시된 LexiconScorer 반환."""
    global _lex
    if _lex is None:
        csv_path = _resolve_csv_path()
        _lex = LexiconScorer.from_csv(
            str(csv_path),
            word_col=os.environ.get(ENV_WORD_COL),
            score_col=os.environ.get(ENV_SCORE_COL),
            debug=_dbg_on(),
        )
    return _lex


def reload_lexicon(path: Optional[str] = None) -> LexiconScorer:
    """CSV를 다시 읽어 전역 인스턴스를 재생성."""
    global _lex
    csv_path = Path(path).expanduser().resolve() if path else _resolve_csv_path()
    _lex = LexiconScorer.from_csv(
        str(csv_path),
        word_col=os.environ.get(ENV_WORD_COL),
        score_col=os.environ.get(ENV_SCORE_COL),
        debug=_dbg_on(),
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
    "reload_lexicon",
    "sincerity_score",
]
