from fastapi import APIRouter, Form
from core.scoring import pre_pipeline

router = APIRouter()

@router.post("/")
async def analyze_text(
    title: str = Form(""),
    content: str = Form(...),
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: float = Form(0.3),
):
    text = f"{title}\n\n{content}".strip()
    result = pre_pipeline(text, denom_mode, w_acc, w_sinc, gate)
    return result
