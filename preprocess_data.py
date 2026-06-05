import pandas as pd
import os
import time
import requests
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(os.getcwd(), "data", "processed", "tripadvisor_30k.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "tripadvisor_preprocessed.csv")
GEO_CACHE_PATH = os.path.join(BASE_DIR, "data", "geocache.json")

def detect_novelty(text: str) -> str:
    if not isinstance(text, str):
        return "boredom alleviation"
    t = text.lower()
    relax_keys = ["relax", "calm", "peaceful", "serene", "spa", "unwind", "chill", "soothing", "quiet", "beach", "lazy", "rest", "massage", "wellness", "peace"]
    arousal_keys = ["thrill", "adrenaline", "exciting", "adventure", "rollercoaster", "zipline", "safari", "party", "nightlife", "club", "arousal", "forest"]
    experience_keys = ["authentic", "immersive", "experience", "local", "culture", "cuisine", "museum", "history", "heritage", "architecture", "explor", "explore"]
    boredom_keys = ["escape", "routine", "different", "change", "new", "break", "variety", "avoid", "boredom", "bored"]
    
    if "forest" in t: return "arousal"
    if "peace" in t: return "relaxation"
    if "bored" in t: return "boredom alleviation"
    if "explor" in t: return "experience"

    if any(k in t for k in relax_keys):
        return "relaxation"
    if any(k in t for k in arousal_keys):
        return "arousal"
    if any(k in t for k in experience_keys):
        return "experience"
    if any(k in t for k in boredom_keys):
        return "boredom alleviation"
    return "boredom alleviation"

def _load_geo_cache() -> dict:
    try:
        if os.path.exists(GEO_CACHE_PATH):
            with open(GEO_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_geo_cache(cache: dict):
    try:
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        with open(GEO_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def geocode_query(query: str, cache: dict) -> dict:
    q = query.strip()
    if not q:
        return None
    if q in cache:
        return cache[q]
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": "novelty-seeking-app"}
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                item = data[0]
                result = {"lat": float(item["lat"]), "lon": float(item["lon"])}
                cache[q] = result
                return result
    except Exception as e:
        print(f"Geocoding error for '{q}': {e}")
        pass
    return None

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Detecting novelty for all reviews...")
    df['novelty_type'] = df['review_text'].apply(detect_novelty)

    print("Geocoding all unique places...")
    geo_cache = _load_geo_cache()
    unique_places = df[['place_name', 'city', 'country']].drop_duplicates().head(500)
    
    latitudes = []
    longitudes = []

    for index, row in unique_places.iterrows():
        query = f"{row['place_name']}, {row['city']}, {row['country']}"
        coords = geocode_query(query, geo_cache)
        if coords:
            latitudes.append(coords['lat'])
            longitudes.append(coords['lon'])
        else:
            latitudes.append(None)
            longitudes.append(None)
        time.sleep(1) # To respect Nominatim's usage policy

    unique_places['lat'] = latitudes
    unique_places['lon'] = longitudes

    print("Merging geocoded data back into the main dataframe...")
    df = pd.merge(df, unique_places, on=['place_name', 'city', 'country'], how='left')

    print(f"Saving pre-processed data to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("Saving updated geocache...")
    _save_geo_cache(geo_cache)

    print("Pre-processing complete!")

if __name__ == "__main__":
    main()
