# src/evaluate.py
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from src.model import BertBiGRU
from src.dataset import ReviewDataset
from transformers import BertTokenizer, BertModel
try:
    from config import *
except ImportError:
    from config import *
try:
    from src.utils import calculate_metrics
except ImportError:
    from utils import calculate_metrics

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    data_path = "data/processed/tripadvisor_30k.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    if not {"review_text", "rating"}.issubset(df.columns):
        print("Dataset missing required columns: 'review_text' and 'rating'.")
        return

    df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
    texts = df["review_text"].values
    labels = df["label"].values

    # Tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    dataset = ReviewDataset(texts, labels, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Pretrained BERT
    bert_base = BertModel.from_pretrained(
        "bert-base-uncased",
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    model = BertBiGRU(bert_model=bert_base).to(device)

    # Load trained weights
    model_path = "models/bert_bigru_model.pt"
    if not os.path.exists(model_path):
        print("Model file not found. Train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()