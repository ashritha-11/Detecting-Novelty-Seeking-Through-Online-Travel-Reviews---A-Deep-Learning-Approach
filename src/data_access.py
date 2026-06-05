import os
import pandas as pd
from typing import Dict, List, Optional
import json
import requests

# ================= PATHS =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "tripadvisor_30k.csv")
GEO_CACHE_PATH = os.path.join(BASE_DIR, "data", "geocache.json")

# ================= LOAD DATA =================
def load_dataset():
    # If the background preprocess script has finished generating the full cached CSV, use it natively
    pre_path = os.path.join(BASE_DIR, "data", "processed", "tripadvisor_preprocessed.csv")
    if os.path.exists(pre_path):
        try:
            return pd.read_csv(pre_path)
        except Exception:
            pass

    # Otherwise, fallback instantly to the dynamic pipeline logic matching exact review text
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found:", DATA_PATH)
        return None

    try:
        df = pd.read_csv(DATA_PATH)
        
        # Pull the exact same function logic that processes the real dataset natively
        import sys
        if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
        from preprocess_data import detect_novelty
        
        # Apply novelty directly from review text rather than generic genre
        df["novelty_type"] = df["review_text"].astype(str).apply(detect_novelty)
        return df

    except Exception as e:
        print("❌ Error loading dataset:", e)
        return None

# ================= TOP PLACES =================
def get_places_summary(limit: int = 10) -> List[Dict]:
    df = load_dataset()
    if df is None:
        return []

    grouped = df.groupby(
        ["place_id", "place_name", "city", "country"],
        as_index=False
    ).agg(
        avg_rating=("rating", "mean"),
        review_count=("review_text", "count")
    ).sort_values(by="review_count", ascending=False)

    return grouped.head(limit).to_dict(orient="records")

# ================= NOVELTY FILTER =================
def places_by_novelty(novelty, limit=10):
    df = load_dataset()
    if df is None:
        return []

    novelty = novelty.strip().lower()

    # Filter by novelty_type, not genre
    filtered = df[df["novelty_type"].str.lower() == novelty]

    if filtered.empty:
        return []

    grouped = filtered.groupby(
        ["place_id", "place_name", "city", "country"],
        as_index=False
    ).agg(
        avg_rating=("rating", "mean"),
        review_count=("review_text", "count")
    ).sort_values(by="review_count", ascending=False)

    return grouped.head(limit).to_dict(orient="records")

# ================= SEARCH =================
def search_places(query: str, limit: int = 20) -> List[Dict]:
    df = load_dataset()
    if df is None:
        return []

    filtered = df[df["review_text"].str.contains(query, case=False, na=False)]

    if filtered.empty:
        return []

    grouped = filtered.groupby(
        ["place_id", "place_name", "city", "country"],
        as_index=False
    ).agg(
        avg_rating=("rating", "mean"),
        review_count=("review_text", "count")
    ).sort_values(by="review_count", ascending=False)

    return grouped.head(limit).to_dict(orient="records")

# ================= PLACE DETAILS =================
def get_place_reviews(place_id: str, city: str, limit: int = 10) -> List[Dict]:
    df = load_dataset()
    if df is None:
        return []

    sub = df[(df["place_id"] == place_id) & (df["city"] == city)]

    if sub.empty:
        return []

    return sub.head(limit).to_dict(orient="records")

def get_place_detail(place_id: str, city: str) -> Optional[Dict]:
    df = load_dataset()
    if df is None:
        return None

    sub = df[(df["place_id"] == place_id) & (df["city"] == city)]

    if sub.empty:
        return None

    row = sub.iloc[0]

    # Dynamically fetch coordinates cache or OSM Nominatim
    lat, lon = None, None
    # Since places are generic ("Attraction X"), querying them directly causes Nominatim to panic and return random places like Amsterdam.
    # We query only the city and country to ensure the map always loads the CORRECT actual location region.
    query = f"{row['city']}, {row['country']}"
    coords = geocode_query(query)
    if coords:
        lat, lon = coords["lat"], coords["lon"]
    else:
        # Fallback coordinates at any cost for every single dataset city
        fallbacks = {
            "New York": (40.7128, -74.0060),
            "Los Angeles": (34.0522, -118.2437),
            "London": (51.5074, -0.1278),
            "Paris": (48.8566, 2.3522),
            "Rome": (41.9028, 12.4964),
            "Sydney": (-33.8688, 151.2093),
            "Mumbai": (19.0760, 72.8777),
            "Istanbul": (41.0082, 28.9784),
            "Dubai": (25.2048, 55.2708),
            "Berlin": (52.5200, 13.4050),
            "Amsterdam": (52.3676, 4.9041),
            "Cape Town": (-33.9249, 18.4241),
            "San Francisco": (37.7749, -122.4194),
            "Toronto": (43.6532, -79.3832),
            "Tokyo": (35.6762, 139.6503)
        }
        lat, lon = fallbacks.get(str(row['city']), (0.0, 0.0))

    # Fetch relevant Image based on city and context (across all reviews for this place)
    city_str = str(row['city'])
    place_name_str = str(row['place_name'])
    
    # Combine all review texts for this specific place to find keywords
    all_reviews_text = " ".join(sub["review_text"].astype(str).tolist())
    
    image_url = get_city_image(city_str, place_name_str, all_reviews_text)

    return {
        "place_id": row["place_id"],
        "place_name": row["place_name"],
        "city": row["city"],
        "country": row["country"],
        "avg_rating": sub["rating"].mean(),
        "lat": lat,
        "lon": lon,
        "image_url": image_url
    }

# ================= IMAGE FETCHING =================
def get_city_image(city: str, place_name: str = "", review_text: str = "") -> str:
    # 1. Check for specific keywords in place_name or review_text to override generic city images
    combined_text = (place_name + " " + review_text).lower()
    
    if "beach" in combined_text or "ocean" in combined_text or "sea" in combined_text:
        return "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=800&auto=format&fit=crop"
    elif "forest" in combined_text or "woods" in combined_text or "jungle" in combined_text:
        return "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?q=80&w=800&auto=format&fit=crop"
    elif "mountain" in combined_text or "peak" in combined_text or "hiking" in combined_text:
        return "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?q=80&w=800&auto=format&fit=crop"
    elif "museum" in combined_text or "art" in combined_text or "gallery" in combined_text:
        return "https://images.unsplash.com/photo-1518998053901-5348d3961a04?q=80&w=800&auto=format&fit=crop"
    elif "park" in combined_text or "garden" in combined_text:
        return "https://images.unsplash.com/photo-1585938389612-a552a28d6914?q=80&w=800&auto=format&fit=crop"
    elif "temple" in combined_text or "shrine" in combined_text or "church" in combined_text:
        return "https://images.unsplash.com/photo-1548013146-72479768bbaa?q=80&w=800&auto=format&fit=crop"

    # 2. Fallback to city-specific overrides
    overrides = {
        "Paris": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques_000222.jpg/800px-La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques_000222.jpg",
        "New York": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/View_of_Empire_State_Building_from_Campanile_top_of_the_Rock.jpg/800px-View_of_Empire_State_Building_from_Campanile_top_of_the_Rock.jpg",
        "Rome": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Colosseum_in_Rome%2C_Italy_-_April_2007.jpg/800px-Colosseum_in_Rome%2C_Italy_-_April_2007.jpg",
        "Tokyo": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Skyscrapers_of_Shinjuku_2009_January.jpg/800px-Skyscrapers_of_Shinjuku_2009_January.jpg",
        "Mumbai": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Mumbai_Aug_2018_%2843397784544%29.jpg/800px-Mumbai_Aug_2018_%2843397784544%29.jpg",
        "London": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall_%28cropped%29.jpg/800px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall_%28cropped%29.jpg",
        "Sydney": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Sydney_Australia._%287848656206%29.jpg/800px-Sydney_Australia._%287848656206%29.jpg",
        "Dubai": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Dubai_Skyscrapers.jpg/800px-Dubai_Skyscrapers.jpg",
        "Istanbul": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hagia_Sophia_Mars_2013.jpg/800px-Hagia_Sophia_Mars_2013.jpg",
        "Berlin": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Brandenburger_Tor_abends.jpg/800px-Brandenburger_Tor_abends.jpg",
        "Amsterdam": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/KeizersgrachtReguliersgrachtAmsterdam.jpg/800px-KeizersgrachtReguliersgrachtAmsterdam.jpg",
        "Cape Town": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Cape_Town_Montage.png/800px-Cape_Town_Montage.png",
        "San Francisco": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/800px-GoldenGateBridge-001.jpg",
        "Toronto": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Toronto_Skyline_Summer_2020.jpg/800px-Toronto_Skyline_Summer_2020.jpg",
        "Los Angeles": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/LA_Skyline_Mountains2.jpg/800px-LA_Skyline_Mountains2.jpg"
    }
    if city in overrides:
        return overrides[city]
        
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "pageimages",
            "format": "json",
            "piprop": "original",
            "titles": city
        }
        headers = {"User-Agent": "NoveltyApp/1.0 (contact@example.com)"}
        r = requests.get(url, params=params, headers=headers, timeout=3)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if "original" in page_data:
                    return page_data["original"]["source"]
    except:
        pass
    
    return "https://images.unsplash.com/photo-1449844908441-8829872d2607?w=800&q=80"

# ================= GEO CACHE (OPTIONAL) =================
def _load_geo_cache():
    if os.path.exists(GEO_CACHE_PATH):
        try:
            with open(GEO_CACHE_PATH, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def _save_geo_cache(cache):
    try:
        with open(GEO_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except:
        pass

def geocode_query(query: str):
    cache = _load_geo_cache()

    if query in cache:
        return cache[query]

    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 1}
        headers = {"User-Agent": "novelty-app"}

        r = requests.get(url, params=params, headers=headers, timeout=5)

        if r.status_code == 200:
            data = r.json()
            if data:
                result = {
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"])
                }
                cache[query] = result
                _save_geo_cache(cache)
                return result
    except:
        pass

    return None