# preprocess_baseline.py
import pandas as pd
import re
from tqdm import tqdm

def normalize_dialogue(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_summary(text):
    # 불필요한 문장 제거
    s = text.strip()
    s = s.replace("\n", " ")

    # 중복 공백 축소
    s = re.sub(r"\s+", " ", s)

    # 문장 마지막 마침표 강제
    if not s.endswith("."):
        s += "."

    return s

def make_dataset():
    df_train = pd.read_csv("data/train.csv")
    df_dev = pd.read_csv("data/dev.csv")

    # Dialogue 전처리
    df_train["dialogue"] = df_train["dialogue"].apply(normalize_dialogue)
    df_dev["dialogue"] = df_dev["dialogue"].apply(normalize_dialogue)

    # Summary baseline 정답 스타일로 유지
    df_train["summary"] = df_train["summary"].apply(normalize_summary)
    df_dev["summary"] = df_dev["summary"].apply(normalize_summary)

    df_train.to_csv("train_prepared.csv", index=False)
    df_dev.to_csv("dev_prepared.csv", index=False)
    print("✓ train_prepared.csv, dev_prepared.csv 생성 완료")

if __name__ == "__main__":
    make_dataset()
