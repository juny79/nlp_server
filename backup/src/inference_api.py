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
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 3]
    return sents


##########################################################
# 2) 불필요 문장 제거 (Solar 출력 정제)
##########################################################
def filter_sentences(sents):
    ban_patterns = [
        r"요약", r"핵심", r"정리", r"주요", r"간결",
        r"다음", r"선택", r"필요한 경우", r"추가",
        r"결론", r"요약하면", r"^[-•*]", r"^\d+\.",
        r"<.*?>", r"^#", r"'''", r'"""', r"\*{2}",
        r"—", r"---"
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
# 3) 원문 기반 단어 매칭 점수 계산
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
# 4) 최종 요약 (Solar → 문장 필터링 → 상위 1~2문장)
##########################################################
def final_summary(raw_summary, dialogue):
    sents = split_sentences(raw_summary)
    sents = filter_sentences(sents)

    if not sents:
        return raw_summary.strip()

    scored = score_sentences(sents, dialogue)
    best_sents = [s for _, s in scored[:2]]
    result = " ".join(best_sents)

    if len(result) > 150:
        result = result[:150].strip()

    return result


##########################################################
# 5) 대회 정답 스타일로 자동 변환 (핵심 기능)
##########################################################
def normalize_to_competition_style(summary, dialogue):

    # 1) 첫 문장만 사용
    first_sentence = re.split(r"[.!?]", summary)[0].strip()
    if len(first_sentence) < 5:
        first_sentence = summary
    s = first_sentence

    # 2) 공손체 → 평서체 현재형으로 변환
    s = re.sub(r"(했습니다|했다|했어요|했어|합니다|합니 다)", "한다", s)
    s = re.sub(r"(되었습니다|됐습니다|됐다)", "된다", s)

    # 3) 불필요 접속사 제거
    s = re.sub(r"(그러나|하지만|결국|한편|또는)", "그리고", s)

    # 4) 인물명(#PersonX#) 강제 정규화
    persons = re.findall(r"(Person\d+)", dialogue)
    persons = sorted(set(persons))

    # Solar가 이름 생략하면 자동 복원
    if persons:
        appeared = [p for p in persons if p in s]
        if len(appeared) == 0 and len(persons) >= 2:
            s = f"#{persons[0]}#와 #{persons[1]}#는 " + s

    s = re.sub(r"(Person\d+)", r"#\1#", s)

    # 5) 문장 길이 제한
    if len(s) > 140:
        s = s[:140].strip()

    # 6) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()

    return s


##########################################################
# 6) Solar API 호출
##########################################################
def call_solar(dialogue):

    prompt_template = f"""
아래의 대화를 **한 문장 또는 두 문장으로만** 매우 간결하게 요약하세요.
불필요한 문구(요약:, 핵심:, 정리:, 번호, 마크다운 등) 없이 사실만 간단히 적으세요.
#PersonX# 표기는 반드시 그대로 유지하세요.

[대화]
{dialogue}

[요약]
"""

    headers = {
        "Authorization": f"Bearer {SOLAR_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "solar-mini",
        "messages": [
            {"role": "system", "content": "당신은 한국어 요약 전문가입니다."},
            {"role": "user", "content": prompt_template}
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    try:
        res = requests.post(SOLAR_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Solar API Error:", e)
        return ""


##########################################################
# 7) 전체 추론 파이프라인
##########################################################
def run_inference(
    data_path="./data/test.csv",
    save_path="./prediction/output.csv"
):
    df = pd.read_csv(data_path)
    summaries = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dialogue = row["dialogue"]

        # 1) Solar API 요약
        raw_summary = call_solar(dialogue)

        # 2) 정제 요약 (불필요 문장 제거)
        clean_summary = final_summary(raw_summary, dialogue)

        # 3) 대회 정답 스타일로 자동 변환 (핵심)
        clean_summary = normalize_to_competition_style(clean_summary, dialogue)

        summaries.append(clean_summary)
        time.sleep(0.4)  # RPM 제한 방지

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
