# backend/routers/posts.py

from fastapi import APIRouter
from core.scoring import _score_extras_with_comments
from core.simulate_chain import USE_DB
from utils.jsonl import _jsonl_list  # jsonl 불러오기 함수

router = APIRouter()

@router.get("/posts")
def list_posts(limit: int = 20, offset: int = 0):
    items_raw = _jsonl_list(limit=limit, offset=offset)
    items = []
    for obj in items_raw:
        sc = obj.get("scores", {}) or {}
        meta = obj.get("meta", {}) or {}
        extras = _score_extras_with_comments(sc, meta)
        items.append({
            "id": int(obj["id"]),
            "title": obj["title"],
            "content": obj.get("content"),
            "created_at": obj.get("created_at"),
            "S_pre": sc.get("S_pre"),
            "S_sinc": sc.get("S_sinc"),
            "S_acc": sc.get("S_acc") or sc.get("S_fact"),
            "gate": obj.get("gate"),
            "gate_pass": sc.get("gate_pass"),
            "S_effective": extras.get("S_effective"),
            "likes": meta.get("likes"),
        })
    return {"ok": True, "items": items, "count": len(items)}
