# backend/routers/analyze.py

from fastapi import APIRouter, Form
from core.scoring import run_scoring_pipeline
from core.simulate_chain import sim_mint

router = APIRouter()

@router.post("/analyze")
async def analyze(
    title: str = Form(...),
    content: str = Form(...),
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: float = Form(0.3)
):
    result = run_scoring_pipeline(
        content,
        denom_mode=denom_mode,
        w_acc=w_acc,
        w_sinc=w_sinc,
        gate=gate
    )

    # 민팅 조건 판단
    minted = False
    tx_hash = None
    token_id = None
    if result["gate_pass"]:
        try:
            tx_hash, token_id = sim_mint("user_address")
            minted = True
        except Exception:
            pass

    return {
        "ok": True,
        "final_score": result["S_pre_raw"],
        "scores": result,
        "minted": minted,
        "tx_hash": tx_hash,
        "token_id": token_id,
    }
