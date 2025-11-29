import pandas as pd
import re
from tqdm import tqdm

def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_dialogue(dialogue):
    lines = dialogue.split("\n")
    speakers = {}
    parsed = []

    for line in lines:
        if ":" not in line:
            continue

        spk, utt = line.split(":", 1)
        spk = spk.strip()
        utt = utt.strip()

        if spk not in speakers:
            speakers[spk] = f"#Person{len(speakers)+1}#"

        parsed.append((speakers[spk], utt))

    return parsed, speakers

def extract_key_sentence(parsed):
    # 단순 baseline: 마지막 발화자를 중심으로 핵심 문장을 구성
    if len(parsed) == 0:
        return ""

    last_speaker, last_utt = parsed[-1]

    # 가장 많이 말한 화자 찾기
    freq = {}
    for spk, _ in parsed:
        freq[spk] = freq.get(spk, 0) + 1
    main_speaker = max(freq, key=freq.get)

    key_utt = ""

    # 주요 발화(의견/요청/설명) 패턴 추출
    for spk, utt in reversed(parsed):
        if any(x in utt for x in ["가고 싶", "하고 싶", "원해", "말해", "생각", "알려", "문제", "아파", "힘들"]):
            key_utt = utt
            break
    if key_utt == "":
        key_utt = last_utt

    return main_speaker, key_utt

def baseline_summary(dialogue):
    dialogue = clean_text(dialogue)
    parsed, speakers = parse_dialogue(dialogue)

    if len(parsed) == 0:
        return ""

    main_spk, key_utt = extract_key_sentence(parsed)

    # 템플릿 기반 요약
    summary = f"{main_spk}은 {key_utt}라고 말합니다."

    return summary


def run_baseline(input_path="test.csv", output_path="baseline_output.csv"):
    df = pd.read_csv(input_path)

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), disable=False):
        fname = row["fname"]
        dialogue = row["dialogue"]

        summary = baseline_summary(dialogue)
        results.append([fname, summary])

    out_df = pd.DataFrame(results, columns=["fname", "summary"])
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_baseline(input_path="data/test.csv", output_path="baseline_output.csv")
