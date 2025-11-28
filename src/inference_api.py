import os
import time
import pandas as pd
from tqdm import tqdm
from .utils import load_api_key, call_solar_api
from .prompt import make_prompt

def inference_api(config):
    df = pd.read_csv(os.path.join(config["general"]["data_path"], "test.csv"))

    api_key = load_api_key(config["solar_api"]["api_key_env"])
    base_url = config["solar_api"]["base_url"]
    model = config["solar_api"]["model"]
    template = config["prompt"]["template"]

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row["fname"]
        dialogue = row["dialogue"]

        prompt = make_prompt(dialogue, template)
        summary = call_solar_api(base_url, api_key, model, prompt)

        results.append({"fname": fname, "summary": summary})

        time.sleep(config["inference"]["sleep"])

    out = pd.DataFrame(results)
    os.makedirs(config["general"]["result_path"], exist_ok=True)
    out.to_csv(os.path.join(config["general"]["result_path"], "output.csv"), index=False)
    return out
