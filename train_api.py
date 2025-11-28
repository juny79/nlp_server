import os
import yaml
import pandas as pd
from tqdm import tqdm
from src.utils import load_api_key, call_solar_api
from src.prompt import make_prompt
from src.scoring import rouge_score

def train_api():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(os.path.join(config["general"]["data_path"], "dev.csv"))

    api_key = load_api_key(config["solar_api"]["api_key_env"])
    base_url = config["solar_api"]["base_url"]
    model = config["solar_api"]["model"]
    template = config["prompt"]["template"]

    preds = []
    labels = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        prompt = make_prompt(row["dialogue"], template)
        summary = call_solar_api(base_url, api_key, model, prompt)
        if summary is None:
            summary = ""  # API 에러 시 빈 문자열
        preds.append(summary)
        labels.append(row["summary"])

    print("Validation Scores:", rouge_score(preds, labels))

if __name__ == "__main__":
    train_api()
