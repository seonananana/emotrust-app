# main.py
# FastAPI 진입점 (라우팅만)
from fastapi import FastAPI
from core import scoring, simulate_chain, model

app = FastAPI()

@app.get("/ping")
def ping():
    return {"msg": "pong"}

@app.post("/analyze")
def analyze(payload: dict):
    text = payload.get("content", "")
    gate = float(payload.get("gate", 0.3))
    w_acc = float(payload.get("w_acc", 0.5))
    w_sinc = float(payload.get("w_sinc", 0.5))
    denom_mode = payload.get("denom_mode", "all")

    result = scoring.run_scoring_pipeline(
        text=text,
        denom_mode=denom_mode,
        w_acc=w_acc,
        w_sinc=w_sinc,
        gate=gate
    )

    if result["gate_pass"]:
        user_address = payload.get("address", "user1")
        tx_hash, token_id = simulate_chain.sim_mint(user_address)
        result.update({"minted": True, "tx_hash": tx_hash, "token_id": token_id})
    else:
        result.update({"minted": False})

    return result
