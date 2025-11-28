# inference_api.py
import os
import re
import time
import pandas as pd
import requests
from tqdm import tqdm

SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")
SOLAR_URL = "https://api.upstage.ai/v1/solar/chat/completions"

##########################################################
# 1) 문장 분리
##########################################################
def split_sentences(text):
    # 문장 분리 (마침표/물음표/느낌표 기준)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 3]
    return sents


##########################################################
# 2) 불필요 문장 제거
##########################################################
def filter_sentences(sents):
    ban_patterns = [
        r"요약", r"핵심", r"정리", r"간결", r"주요", r"다음",
        r"선택", r"필요한 경우", r"추가 설명", r"결론적으로",
        r"요약하면", r"^[-•*]", r"^\d+\.", r"<.*?>",
        r"^#", r"'''", r'"""', r"\*{2}", r"^—", r"---"
    ]

    filtered = []
    for s in sents:
        if any(re.search(bp, s, re.IGNORECASE) for bp in ban_patterns):
            continue
        if len(s) < 5:
            continue
        filtered.append(s)

    return filtered


##########################################################
# 3) 원문 대화 기반 단어 매칭 점수 계산
##########################################################
def score_sentences(sents, dialogue):
    dialog_words = set(dialogue.replace("#", "").replace(":", "").split())
    scores = []

    for s in sents:
        words = set(s.split())
        score = len(words & dialog_words) / (len(words) + 1e-6)
        scores.append((score, s))

    scores.sort(reverse=True)
    return scores


##########################################################
# 4) 최종 요약 생성 (1~2 문장)
##########################################################
def final_summary(raw_summary, dialogue):
    # Step 1: 문장 분리
    sents = split_sentences(raw_summary)

    # Step 2: 불필요 문장 제거
    sents = filter_sentences(sents)

    if not sents:
        return ""

    # Step 3: 점수 기반 정렬
    scored = score_sentences(sents, dialogue)

    # Step 4: 상위 1~2문장 선택
    best_sents = [s for _, s in scored[:2]]

    result = " ".join(best_sents)

    # Step 5: 길이 제한 (대회 dev 스타일 길이 맞춤)
    if len(result) > 150:
        result = result[:150].strip()

    return result


##########################################################
# 5) Solar API 요청
##########################################################
def call_solar(dialogue):
    prompt_template = f"""
다음은 일상 대화입니다. 이를 읽고 **한 문장 또는 두 문장으로만** 요약하세요.
불필요한 단어 없이, 사실만 간결하게 적어야 합니다.

[대화]
{dialogue}

[요약]
"""
    headers = {
        "Authorization": f"Bearer {SOLAR_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "solar-mini",
        "messages": [
            {"role": "system", "content": "당신은 한국어 요약 전문가입니다."},
            {"role": "user", "content": prompt_template}
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    try:
        res = requests.post(SOLAR_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Solar API Error:", e)
        return ""


##########################################################
# 6) 전체 추론 파이프라인
##########################################################
def run_inference(data_path="./data/test.csv", save_path="./prediction/output.csv"):
    df = pd.read_csv(data_path)
    summaries = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dialogue = row["dialogue"]

        # 1) Solar API 호출
        raw_summary = call_solar(dialogue)

        # 2) 후처리(정제 요약)
        clean_summary = final_summary(raw_summary, dialogue)

        summaries.append(clean_summary)

        time.sleep(0.5)  # RPM 제한 보호

    df_out = pd.DataFrame({
        "fname": df["fname"],
        "summary": summaries
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False)

    print("\n=== Inference Completed ===")
    print("Saved to:", save_path)
    return df_out


##########################################################
# 실행
##########################################################
if __name__ == "__main__":
    run_inference()
