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
from typing import Dict, Optional, Tuple, Iterable, List

# ===== 설정 =====
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
    return os.environ.get(ENV_DEBUG, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _casefold(s: Optional[str]) -> str:
    return (s or "").strip().casefold()


def _pick_col(fieldnames: Iterable[str], user_pref: Optional[str], aliases: set[str]) -> Optional[str]:
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
        class _ExcelTab(csv.Dialect):
            delimiter = "\t"
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\r\n"
            quoting = csv.QUOTE_MINIMAL
        if "\t" in sample and sample.count("\t") >= sample.count(","):
            return _ExcelTab
        return csv.excel


def _candidate_paths() -> List[Path]:
    cands: List[Path] = []
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

    seen, out = set(), []
    for p in cands:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


@dataclass(slots=True)
class LexiconScorer:
    vocab: Dict[str, float]
    path: str
    word_col: str
    score_col: str
    min_v: float = 0.0
    max_v: float = 1.0

    # ---------- 생성자 헬퍼 ----------
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
                    f"Column detection failed. header={reader.fieldnames}\n"
                    f"Need word_col in {_WORD_ALIASES} and score_col in {_SCORE_ALIASES} "
                    f"(or set {ENV_WORD_COL}/{ENV_SCORE_COL})."
                )

            buckets: Dict[str, List[float]] = defaultdict(list)
            row_cnt = 0
            for row in reader:
                row_cnt += 1
                w_raw = (row.get(wcol) or "").strip()
                if not w_raw:
                    continue
                w = ucd.normalize("NFKC", w_raw).casefold()

                sval = ("" if row.get(scol) is None else str(row.get(scol))).strip()
                sval = sval.replace(",", ".")  # 콤마 소수점 허용
                try:
                    s = float(sval)
                except Exception:
                    continue
                if math.isfinite(s):
                    buckets[w].append(s)

        vocab = {w: (sum(vals) / len(vals)) for w, vals in buckets.items() if vals}
        if not vocab:
            raise ValueError(f"No valid rows parsed from CSV: {p} (rows={row_cnt})")

        min_v = min(vocab.values())
        max_v = max(vocab.values())

        if debug or _dbg_on():
            print(f"[lexicon] path={p}")
            print(f"[lexicon] rows={row_cnt}, words={len(vocab)}, min_v={min_v:.4f}, max_v={max_v:.4f}")
            print(f"[lexicon] columns -> word_col='{wcol}', score_col='{scol}'")

        return cls(
            vocab=vocab,
            path=str(p),
            word_col=wcol,
            score_col=scol,
            min_v=min_v,
            max_v=max_v,
        )

    # ---------- 전처리 ----------
    @staticmethod
    def _norm_token(t: str) -> str:
        return ucd.normalize("NFKC", t).casefold()

    def _norm_score(self, x: float) -> float:
        # [0,1] 밖이면 min–max 정규화
        if self.min_v < 0.0 or self.max_v > 1.0:
            if self.max_v == self.min_v:
                return 0.0
            x = (x - self.min_v) / (self.max_v - self.min_v)
        return clamp01(x)

    def _tokenize(self, text: str) -> List[str]:
        return _WORD_RE.findall(text)

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

        matched_scores: List[float] = []
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
    for p in _candidate_paths():
        if p.exists():
            return p
    tried = "\n  - " + "\n  - ".join(str(p) for p in _candidate_paths())
    raise FileNotFoundError(f"Lexicon CSV not found. Tried:{tried}")


def get_lexicon() -> LexiconScorer:
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
    return get_lexicon().sincerity(text, mode=mode, alpha=alpha)


__all__ = [
    "LexiconScorer",
    "get_lexicon",
    "reload_lexicon",
    "sincerity_score",
]
