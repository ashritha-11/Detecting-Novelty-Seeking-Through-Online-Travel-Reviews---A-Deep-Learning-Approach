from src.data_access import places_by_novelty

def test_novelty(n):
    print(f"Testing novelty map: '{n}'")
    places = places_by_novelty(n, limit=50)
    print(f"  Places found: {len(places)}")
    if len(places) > 0:
        print(f"  Sample place: {places[0]['place_name']} (Score: {places[0]['score']})")

test_novelty("relaxation")
test_novelty("Relaxation")
test_novelty("arousal seeking")
test_novelty("experience")
test_novelty("boredom alleviation")
