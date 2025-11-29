import os
import re
import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration
)

############################################################
# 1) Special tokens 설정
############################################################

SPECIAL_TOKENS = [
    "#Person1#", "#Person2#", "#Person3#",
    "#PhoneNumber#", "#Address#", "#PassportNumber#"
]


############################################################
# 2) KoBART 모델 & 토크나이저 로드
############################################################

def load_model(checkpoint_path, model_name="digit82/kobart-summarization"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Special tokens 추가 (KoBART 핵심 문제 해결)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # 모델 로드
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)

    # 토크나이저 추가 토큰 수만큼 임베딩 재확장
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


############################################################
# 3) 전처리: 공백, 줄바꿈 정리
############################################################

def clean_dialogue(text):
    if not isinstance(text, str):
        return ""
    t = text.replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


############################################################
# 4) 요약 생성 함수
############################################################

@torch.no_grad()
def generate_summary(model, tokenizer, dialogue):

    inputs = tokenizer(
        dialogue,
        max_length=640,      # 대화 길이에 최적화
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=100,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 불필요한 토큰 제거
    summary = summary.replace("<s>", "").replace("</s>", "").strip()

    return summary


############################################################
# 5) main inference
############################################################

def run_inference(input_csv="data/test.csv", checkpoint="./checkpoints/", output_csv="summary_A_model_output.csv"):

    tokenizer, model = load_model(checkpoint)

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    df = pd.read_csv(input_csv)

    summaries = []

    print("▶ KoBART inference 시작...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        dialogue = clean_dialogue(row["dialogue"])

        # 요약 생성
        summary = generate_summary(model, tokenizer, dialogue)

        # special tokens 유지 체크 (무결성)
        for token in SPECIAL_TOKENS:
            if token in dialogue and token not in summary:
                summary = token + " " + summary

        summaries.append(summary)

    df_out = pd.DataFrame({
        "fname": df["fname"],
        "summary": summaries
    })

    df_out.to_csv(output_csv, index=False)
    print(f"완료: {output_csv} 저장됨")


if __name__ == "__main__":
    run_inference()
