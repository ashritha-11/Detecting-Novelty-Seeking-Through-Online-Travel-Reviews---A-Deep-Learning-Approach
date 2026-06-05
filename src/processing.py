import pandas as pd
import os
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def combine_datasets(input_folder, output_file):
    dfs = []

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_folder, file))

            if "review_text" not in df.columns:
                continue

            df["review_text"] = df["review_text"].apply(clean_text)
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna()

    combined.to_csv(output_file, index=False)
    print("Datasets cleaned & combined successfully!")