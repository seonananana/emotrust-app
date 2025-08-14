# backend/core/model.py

import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class KoBERTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(output.pooler_output).squeeze(1)

def predict_s_acc(text: str) -> float:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = KoBERTRegressor().to(DEVICE)
    model.load_state_dict(torch.load("kobert_regression.pt", map_location=DEVICE))
    model.eval()

    encoded = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        score = model(input_ids, attention_mask).item()
        return max(0.0, min(1.0, score))
