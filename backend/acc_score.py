# acc_score.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import os

# 경로
MODEL_PATH = "kobert_regression.pt"

# 1. Dataset 정의
class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# 2. 모델 정의
class KoBERTRegressionModel(nn.Module):
    def __init__(self):
        super(KoBERTRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        out = self.dropout(pooled)
        return self.regressor(out).squeeze(1)

# 3. 라벨 매핑 (분류 → 회귀 스코어)
def label_to_score(label):
    if label == "긍정":
        return 1.0
    elif label == "중립":
        return 0.5
    elif label == "부정":
        return 0.0
    return 0.5  # fallback

# 4. 학습 함수
def train_model():
    df = pd.read_csv("finance_data.csv")
    df['score'] = df['label'].apply(label_to_score)

    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    dataset = TextRegressionDataset(df['sentence'].tolist(), df['score'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = KoBERTRegressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            targets = batch['target']
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ KoBERT 회귀모델 저장 완료:", MODEL_PATH)

# 5. 예측 함수
def predict_score(text: str) -> float:
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = KoBERTRegressionModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=64)
    with torch.no_grad():
        output = model(encoding['input_ids'], encoding['attention_mask'])
    return float(torch.sigmoid(output).item())  # 0~1 범위 보정

# 6. analyzer에서 호출할 함수
def compute_accuracy_score(text: str) -> dict:
    score = predict_score(text)
    return {
        "S_acc": score,
        "method": "kobert-regression"
    }

# 직접 학습 시
if __name__ == "__main__":
    train_model()
