import yaml
from src.inference_api import inference_api

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    out = inference_api(config)
    print(out.head())
