import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from src.model import BertBiGRU

print("[SUCCESS] NEW PREDICT FILE LOADED")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bert_bigru_model.pt")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ IMPORTANT: num_classes = 4
model = BertBiGRU(num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

LABEL_MAP = {
    0: "experience",
    1: "arousal",
    2: "relaxation",
    3: "boredom alleviation"
}

def predict(text):
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        label = torch.argmax(probs, dim=1).item()

    return LABEL_MAP[label]