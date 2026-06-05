import pandas as pd
import os

filepath = os.path.join(os.path.dirname(__file__), "data", "processed", "tripadvisor_30k.csv")

def revert_name(row):
    pid = str(row['place_id'])
    city = str(row['city'])
    try:
        num = int(pid.replace('P', ''))
        # Original was exactly something like "Attraction 19 in Paris"? Wait, maybe it was just "Attraction 19 in Par" 
        # because of truncated console output earlier? No, the dataframe column had "Attraction 19 in Paris". We'll just restore exactly that.
        if city == "New York":
            return f"Attraction {num} in New York"
        return f"Attraction {num} in {city}"
    except Exception:
        return f"Attraction in {city}"

def main():
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    df['place_name'] = df.apply(revert_name, axis=1)
    
    # ensure it writes back successfully
    df.to_csv(filepath, index=False)
    print("Reverted place names successfully!")

if __name__ == "__main__":
    main()
