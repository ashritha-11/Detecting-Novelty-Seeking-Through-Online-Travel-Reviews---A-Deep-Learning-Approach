from src.data_access import genre_places, genre_reviews
import pandas as pd

def test_genre(g):
    print(f"Testing genre: '{g}'")
    places = genre_places(g, limit=50)
    reviews = genre_reviews(g, limit=100)
    print(f"  Places found: {len(places)}")
    print(f"  Reviews found: {len(reviews)}")
    if len(places) > 0:
        print(f"  Sample place: {places[0]['place_name']}")

test_genre("relaxation")
test_genre("Relaxation")
test_genre("RELAXATION")
test_genre("arousal seeking")
test_genre("experience")
test_genre("boredom alleviation")
