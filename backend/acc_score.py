# backend/acc_score.py
import os, math, torch
from transformers import AutoTokenizer, AutoModel

class KobertScorer:
    def __init__(self):
        self.model_name = os.getenv("KOBERT_MODEL", "skt/kobert-base-v1")
        self.device = os.getenv("KOBERT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = os.getenv("KOBERT_POOLING", "mean")
        self.temperature = float(os.getenv("KOBERT_TEMPERATURE", "1.0"))
        self.head_path = os.getenv("KOBERT_HEAD_PATH")  # 없으면 A안 OFF
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        for p in self.backbone.parameters(): p.requires_grad = False
        self.head = None
        if self.head_path and os.path.exists(self.head_path):
            self.head = torch.nn.Linear(768, 1).to(self.device)
            self.head.load_state_dict(torch.load(self.head_path, map_location=self.device))
            self.head.eval()

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        if not text: return torch.zeros(768, device=self.device)
        batch = self.tok(text, return_tensors="pt", truncation=True, max_length=256)
        batch = {k:v.to(self.device) for k,v in batch.items()}
        out = self.backbone(**batch)
        if self.pooling == "cls":
            v = out.last_hidden_state[:,0,:]
        else:
            m = batch["attention_mask"].unsqueeze(-1)
            v = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1)
        return v.squeeze(0)

    @torch.inference_mode()
    def score_text(self, text: str) -> Optional[float]:
        if self.head is None:
            return None  # 헤드 없으면 A안 비활성
        v = self.encode(text)
        logit = self.head(v.unsqueeze(0)).squeeze(0)
        T = max(1e-3, self.temperature)
        return float(torch.sigmoid(logit / T).item())

_sc: Optional[KobertScorer] = None
def score_text(text: str) -> Optional[float]:
    global _sc
    if _sc is None: _sc = KobertScorer()
    return _sc.score_text(text)
