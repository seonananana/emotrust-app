from fastapi import APIRouter
from core.simulate_chain import sim_mint

router = APIRouter()

posts_db = []

@router.get("/")
def list_posts():
    return {"ok": True, "items": posts_db, "count": len(posts_db)}

@router.post("/mint")
def analyze_and_mint(post: dict):
    if post.get("gate_pass"):
        tx_hash, token_id = sim_mint(post.get("to_address", "demo"))
        post["minted"] = True
        post["tx_hash"] = tx_hash
        post["token_id"] = token_id
    posts_db.append(post)
    return {"ok": True, "minted": post.get("minted", False), "post": post}
