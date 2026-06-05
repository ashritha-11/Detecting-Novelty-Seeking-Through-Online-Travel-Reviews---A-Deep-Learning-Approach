import os
import torch
from src.model import BertBiGRU

os.makedirs("models", exist_ok=True)

model = BertBiGRU()
torch.save(model.state_dict(), "models/bert_bigru_model.pt")

print("Fresh model file created successfully!")
print("File size:", os.path.getsize("models/bert_bigru_model.pt"), "bytes")