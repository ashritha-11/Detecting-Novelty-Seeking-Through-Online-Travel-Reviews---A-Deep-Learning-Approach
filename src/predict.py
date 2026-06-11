import os
import torch
import torch.nn.functional as F
import gdown
from transformers import BertTokenizer
from src.model import BertBiGRU

print("[SUCCESS] NEW PREDICT FILE LOADED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "bert_bigru_model.pt")

# =====================================================
# GOOGLE DRIVE DOWNLOAD
# =====================================================

# Replace with your Google Drive File ID
FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"

DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    print("[SUCCESS] Model downloaded successfully!")

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =====================================================
# MODEL
# =====================================================
model = BertBiGRU(num_classes=4).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print("[SUCCESS] Model Loaded")

# =====================================================
# LABELS
# =====================================================
LABEL_MAP = {
    0: "experience",
    1: "arousal",
    2: "relaxation",
    3: "boredom alleviation"
}

# =====================================================
# PREDICTION FUNCTION
# =====================================================
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

        outputs = model(
            input_ids,
            attention_mask
        )

        probs = F.softmax(outputs, dim=1)

        predicted_class = torch.argmax(
            probs,
            dim=1
        ).item()

    return LABEL_MAP[predicted_class]
