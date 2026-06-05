from src.data_access import genre_places, genre_reviews, detect_genre, detect_novelty
import pandas as pd

print("Testing genre_places('relaxation')...")
places = genre_places("relaxation")
print(f"Found {len(places)} places.")

print("Testing genre_reviews('relaxation')...")
reviews = genre_reviews("relaxation")
print(f"Found {len(reviews)} reviews.")

print("Testing detect_novelty on a relaxation text...")
text = "This was such a relax and calm beach. Very peaceful and quiet. I enjoyed the spa."
print(f"detect_novelty says: {detect_novelty(text)}")
print(f"detect_genre says: {detect_genre(text)}")

print("Testing detect_novelty on the whole dataset...")
DATA_PATH = "data/processed/tripadvisor_30k.csv"
df = pd.read_csv(DATA_PATH)
df["genre_computed"] = df["review_text"].astype(str).apply(detect_novelty)
print(df["genre_computed"].value_counts())
