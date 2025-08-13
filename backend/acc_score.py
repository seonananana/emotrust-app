import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------------------
# 설정
# ---------------------------
CSV_PATH = "data/finance_data.csv"
MODEL_PATH = "kobert_regression.pt"
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 2  # EarlyStopping 기준

# ---------------------------
# 데이터셋 정의
# ---------------------------
class FinanceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_map):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['kor_sentence'])
        label_str = self.data.iloc[idx]['labels']
        label = self.label_map[label_str]

        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ---------------------------
# 모델 정의
# ---------------------------
class KoBERTRegressor(nn.Module):
    def __init__(self):
        super(KoBERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.regressor(cls_output).squeeze(1)

# ---------------------------
# 학습 함수 with EarlyStopping
# ---------------------------
def train():
    df = pd.read_csv(CSV_PATH, encoding="cp949")

    label_map = {
        'negative': 0.0,
        'neutral': 0.5,
        'positive': 1.0
    }

    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_set = FinanceDataset(train_df, tokenizer, label_map)
    val_set = FinanceDataset(val_df, tokenizer, label_map)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)

    model = KoBERTRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {total_loss:.4f}")

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        # EarlyStopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ 모델 저장됨 (성능 향상)")
        else:
            patience_counter += 1
            print(f"⚠️ EarlyStopping 카운트: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("🛑 EarlyStopping: 성능 향상 없음 → 학습 종료")
                break

# ---------------------------
# 추론 함수
# ---------------------------
def predict_with_kobert(text: str) -> float:
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    model = KoBERTRegressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=64
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        score = model(input_ids, attention_mask).item()
        return max(0.0, min(1.0, score))  # clamp

# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    train()
