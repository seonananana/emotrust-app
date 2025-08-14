import torch
from transformers import BertTokenizer
from kobert_tokenizer import KoBERTTokenizer
from torch.nn.functional import softmax

# 모델 클래스 정의는 기존 main_backup.py에서 가져와야 함
class KoBERTRegressor(torch.nn.Module):
    def __init__(self, bert_model, hidden_size=768):
        super(KoBERTRegressor, self).__init__()
        self.bert = bert_model
        self.regressor = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.regressor(cls_output)

# 모델과 토크나이저 로딩
tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
bert_model = torch.hub.load('SKTBrain/KoBERT', 'kobert_pytorch_model')
model = KoBERTRegressor(bert_model)
model.load_state_dict(torch.load("kobert_regression.pt", map_location=torch.device("cpu")))
model.eval()

def score_accuracy(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    score = output.item()
    return round(score, 4)
