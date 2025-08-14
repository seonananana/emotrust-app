# backend/utils/preprocess.py

import re

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text
