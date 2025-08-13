import re
import unicodedata
from typing import List

# --- 전처리 (URL/제로폭/공백/소문자/NFKC) ---

_url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# 제로폭/비가시 공백 문자 세트 확장: ZWSP/ZWJ/ZWNJ/WORD JOINER/BOM/NBSP 등
ZW_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00A0\u202F]")

def preprocess_text(text: str) -> str:
    """URL/제로폭/공백 정리 + lower + NFKC 정규화"""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _url_re.sub(" ", t)
    t = ZW_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t
