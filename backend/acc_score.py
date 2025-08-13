import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL_PATH = "kobert_regression.pt"
CSV_PATH = "backend/data/finance_data.csv"

# 1. Dataset class
class FinanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

# 2. Regression Model
class KoBERTRegressor(nn.Module):
    def __init__(self):
        super(KoBERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] representation
        return self.regressor(pooled).squeeze(1)  # [batch_size]

# 3. 학습 함수
def train_model():
    df = pd.read_csv(CSV_PATH)

    # 감성 → 정확성 점수 매핑
    mapping = {"부정": 0.0, "중립": 0.5, "긍정": 1.0}
    df = df[df["label"].isin(mapping.keys())]  # 유효한 라벨만
    df["score"] = df["label"].map(mapping)

    tokenizer = BertTokenizer.from_pretrained("skt/kobert-base-v1")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["score"].tolist(), test_size=0.1, random_state=42
    )

    train_dataset = FinanceDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinanceDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = KoBERTRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# 4. 예측 함수 (사용자 입력용)
def predict_accuracy(text):
    tokenizer = BertTokenizer.from_pretrained("skt/kobert-base-v1")
    model = KoBERTRegressor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(
            encoded["input_ids"],
            encoded["attention_mask"]
        )
        score = output.item()
        return max(0.0, min(1.0, round(score, 4)))

# 5. CLI 테스트용 (선택)
if __name__ == "__main__":
    train_model()
    print("✅ 테스트 결과: ", predict_accuracy("이 회사는 올해 매출이 증가할 것이다."))
