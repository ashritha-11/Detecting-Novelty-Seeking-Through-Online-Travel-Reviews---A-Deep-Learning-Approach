import pandas as pd
import os

def detect_novelty(text: str) -> str:
    if not isinstance(text, str):
        return "boredom alleviation"
    t = text.lower()
    relax_keys = ["relax", "calm", "peaceful", "serene", "spa", "unwind", "chill", "soothing", "quiet", "beach", "lazy", "rest", "massage", "wellness"]
    arousal_keys = ["thrill", "adrenaline", "exciting", "adventure", "rollercoaster", "zipline", "safari", "party", "nightlife", "club", "arousal"]
    experience_keys = ["authentic", "immersive", "experience", "local", "culture", "cuisine", "museum", "history", "heritage", "architecture"]
    boredom_keys = ["escape", "routine", "different", "change", "new", "break", "variety", "avoid", "boredom"]
    
    if any(k in t for k in relax_keys):
        return "relaxation"
    if any(k in t for k in arousal_keys):
        return "arousal seeking"
    if any(k in t for k in experience_keys):
        return "experience"
    if any(k in t for k in boredom_keys):
        return "boredom alleviation"
    return "boredom alleviation"

DATA_PATH = "data/processed/tripadvisor_30k.csv"
df = pd.read_csv(DATA_PATH)
df["__novelty"] = df["review_text"].astype(str).apply(detect_novelty)
print(df["__novelty"].value_counts())
