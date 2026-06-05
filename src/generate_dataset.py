
import random
import pandas as pd
from datetime import datetime, timedelta

# ✅ FIX: use current working directory in Colab
BASE_DIR = os.getcwd()
OUT_PATH = os.path.join(BASE_DIR, "tripadvisor_30k.csv")

cities = [
    ("Paris", "France"), ("New York", "USA"), ("London", "UK"), ("Barcelona", "Spain"),
    ("Tokyo", "Japan"), ("Sydney", "Australia"), ("Rome", "Italy"), ("Dubai", "UAE"),
    ("Singapore", "Singapore"), ("Bangkok", "Thailand"), ("Istanbul", "Turkey"),
    ("Berlin", "Germany"), ("Amsterdam", "Netherlands"), ("Toronto", "Canada"),
    ("San Francisco", "USA"), ("Los Angeles", "USA"), ("Cape Town", "South Africa"),
    ("New Delhi", "India"), ("Mumbai", "India"), ("Seoul", "South Korea")
]

# ✅ ONLY 4 TYPES
genres = ["experience", "arousal", "relaxation", "boredom alleviation"]

genre_templates = {
    "experience": [
        "Rich history and inspiring museum exhibits",
        "Beautiful architecture and cultural heritage",
        "Authentic local traditions and art experience"
    ],
    "arousal": [
        "Thrilling adventure and exciting activities",
        "Lively nightlife with music and fun",
        "Adrenaline-filled experience with lots of energy"
    ],
    "relaxation": [
        "Peaceful park with calm surroundings",
        "Serene beach and relaxing atmosphere",
        "Quiet place perfect for unwinding and rest"
    ],
    "boredom alleviation": [
        "Fun place to break routine and try something new",
        "Great escape from daily boredom",
        "Interesting spot with unique experiences"
    ]
}

def random_place(i):
    city, country = random.choice(cities)
    name = f"Attraction {i%200 + 1} in {city}"
    return name, city, country

def generate_rows(n=30000):
    rows = []
    start = datetime(2022, 1, 1)

    for i in range(n):
        place_id = f"P{str(i%200 + 1).zfill(3)}"
        place_name, city, country = random_place(i)
        rating = random.randint(1, 5)

        g = random.choice(genres)
        text = random.choice(genre_templates[g])

        date = start + timedelta(days=random.randint(0, 900))
        reviewer = f"User{random.randint(1, 99999)}"

        rows.append([
            place_id, place_name, city, country,
            rating, text, date.strftime("%Y-%m-%d"),
            reviewer, 0.0, 0.0,
            "https://www.tripadvisor.com/", g
        ])

    return rows

print("⏳ Generating dataset...")
rows = generate_rows(30000)

df = pd.DataFrame(rows, columns=[
    "place_id","place_name","city","country","rating",
    "review_text","review_date","reviewer",
    "latitude","longitude","url","genre"
])

df.to_csv(OUT_PATH, index=False)

print("✅ Dataset generated!")
print("📁 Saved at:", OUT_PATH)
print("📊 Total rows:", len(df))