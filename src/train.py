# src/train.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertModel  # remove AdamW here
from torch.optim import AdamW  # import AdamW from torch
from tqdm import tqdm

# ==========================
# Configuration
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
NUM_CLASSES = 4  # Binary labels

DATA_PATH = "data/processed/tripadvisor_30k.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "bert_bigru_model.pt")

# ==========================
# Load Dataset
# ==========================
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Dataset not found at {DATA_PATH}.")

df = pd.read_csv(DATA_PATH)
if not {"review_text", "rating"}.issubset(df.columns):
    raise SystemExit("Dataset missing required columns: 'review_text' and 'rating'.")

# Binary label: 1 if rating >= 4 else 0
LABEL_MAP = {
    "experience": 0,
    "arousal": 1,
    "relaxation": 2,
    "boredom alleviation": 3
}

df["genre"] = df["genre"].str.strip().str.lower()
df["label"] = df["genre"].map(LABEL_MAP)

texts = df["review_text"].tolist()
labels = df["label"].tolist()
texts = df["review_text"].tolist()
labels = df["ns_label"].tolist()

# Train/Validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")

# ==========================
# Compute Class Weights
# ==========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),  # ✅ must be np.array
    y=np.array(train_labels)
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ==========================
# Tokenizer
# ==========================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ==========================
# Dataset Class
# ==========================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ReviewDataset(train_texts, train_labels)
val_dataset = ReviewDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==========================
# Model: BERT + BiGRU
# ==========================
class BertBiGRU(nn.Module):
    def __init__(self, bert_model=None, hidden_size=128, num_classes=2):
        super(BertBiGRU, self).__init__()
        self.bert = bert_model if bert_model else BertModel.from_pretrained(MODEL_NAME)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        gru_output, _ = self.gru(sequence_output)
        pooled = torch.mean(gru_output, dim=1)
        x = self.dropout(pooled)
        return self.fc(x)

bert_base = BertModel.from_pretrained(MODEL_NAME)
model = BertBiGRU(bert_model=bert_base,num_classes=4).to(DEVICE)

# ==========================
# Optimizer & Loss
# ==========================
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ==========================
# Training Loop
# ==========================
best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []    

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

        loop.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["label"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels_batch.cpu().numpy())

    train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {total_loss/len(train_loader):.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Best model saved!")

print("\nTraining completed!")
print("Best Validation Accuracy:", best_val_acc)