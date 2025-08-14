# backend/utils/file_io.py

import json
import os
from typing import List, Dict, Any

POSTS_FILE = os.path.join("data", "posts.jsonl")

def _ensure_data_dir():
    os.makedirs(os.path.dirname(POSTS_FILE), exist_ok=True)

def save_post_jsonl(data: Dict[str, Any]):
    _ensure_data_dir()
    with open(POSTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_posts_jsonl(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    if not os.path.exists(POSTS_FILE):
        return []
    with open(POSTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-(offset + limit):-offset if offset else None]]
