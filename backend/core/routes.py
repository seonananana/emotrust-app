from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from core.scoring import pre_pipeline, combine_scores
from core.model import score_accuracy
from utils.preprocess import clean_text

router = APIRouter()

@router.post("/analyze")
async def analyze_text(
    title: str = Form(...),
    content: str = Form(...),
    denom_mode: str = Form("all"),
    w_acc: float = Form(0.5),
    w_sinc: float = Form(0.5),
    gate: float = Form(0.12),
):
    try:
        cleaned_text = clean_text(content)
        s_sinc, denom = pre_pipeline(cleaned_text, mode=denom_mode)
        s_acc = score_accuracy(cleaned_text)
        s_pre = combine_scores(s_sinc, s_acc, w_sinc=w_sinc, w_acc=w_acc)
        
        result = {
            "S_sinc": s_sinc,
            "S_acc": s_acc,
            "S_pre": s_pre,
            "denom": denom,
            "gate": gate,
            "result": "통과" if s_pre >= gate else "불가"
        }
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
