# utils.py
import pandas as pd
from preprocess import DialoguePreprocessor

def load_dataset(path):
    df = pd.read_csv(path)
    dp = DialoguePreprocessor()

    df["dialogue"] = df["dialogue"].apply(dp.normalize_dialogue)
    df["summary"] = df["summary"].apply(dp.normalize_summary)

    return df
