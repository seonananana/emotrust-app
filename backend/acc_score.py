import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from kobert_transformers import get_tokenizer, get_kobert_model
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# KoBERT 회귀 모델
class KoBERTRegressor(nn.Module):
    def __init__(self):
        super(KoBERTRegressor, self).__init__()
        self.bert = get_kobert_model()
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.regressor(pooled).squeeze(1)  # (batch,) 형태

# 커스텀 Dataset
class KoBERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df["kor_sentence"].tolist()
        self.labels = df["score"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# 라벨 → 점수 매핑
label2score = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df[df["labels"].isin(label2score)]
    df["score"] = df["labels"].map(label2score)
    return df

def train():
    df = load_data("finance_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = get_tokenizer()
    train_dataset = KoBERTDataset(train_df, tokenizer)
    val_dataset = KoBERTDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = KoBERTRegressor().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "kobert_regressor.pth")
    print("✅ Model saved!")

# 예측
@torch.no_grad()
def predict_score(sentence):
    tokenizer = get_tokenizer()
    model = KoBERTRegressor().cuda()
    model.load_state_dict(torch.load("kobert_regressor.pth"))
    model.eval()

    encoded = tokenizer(sentence, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    score = model(input_ids, attention_mask).item()
    return round(score, 3)

if __name__ == "__main__":
    train()
    # test
    test_sent = "이 회사는 2025년에 높은 수익을 기록했다."
    print("S_acc =", predict_score(test_sent))
