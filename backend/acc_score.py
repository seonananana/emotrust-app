# acc_score.py (KoBERT 회귀 학습 + 추론)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
import os

MODEL_PATH = "kobert_regression.pt"
CSV_PATH = "backend/data/finance_data.csv"

class FinanceDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=64):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # '중립' → 0.5, '긍정' → 1.0, '부정' → 0.0 으로 정규화
        self.label_map = {'중립': 0.5, '긍정': 1.0, '부정': 0.0}
        self.data = self.data[self.data['label'].isin(self.label_map)]
        self.data['score'] = self.data['label'].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]['sentence'])
        label = self.data.iloc[idx]['score']

        inputs = self.tokenizer(sentence,
                                return_tensors="pt",
                                max_length=self.max_len,
                                truncation=True,
                                padding='max_length')

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.float)
        }

class KoBERTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)

# 학습 함수
def train():
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    dataset = FinanceDataset(CSV_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = KoBERTRegressor()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved at", MODEL_PATH)

# 추론 함수
def predict_s_acc(text: str) -> float:
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = KoBERTRegressor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
    with torch.no_grad():
        output = model(inputs['input_ids'], inputs['attention_mask'])
        score = output.item()

    return float(np.clip(score, 0.0, 1.0))

# 학습을 처음에 할 경우만 실행
if __name__ == "__main__":
    train()
