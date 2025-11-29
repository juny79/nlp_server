import pandas as pd
import os
import json
import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

API_KEY = os.getenv("SOLAR_API_KEY")

# 최신 Upstage Solar API endpoint
API_URL = "https://api.upstage.ai/v1/chat/completions"

INPUT_CSV = "summary_A_model_output.csv"
OUTPUT_CSV = "summary_B_solar_output.csv"

DEBUG_MODE = True   # 필요시 False로 변경 가능


# ---------------------------
#  System Prompt (Summary-B)
# ---------------------------
SYSTEM_PROMPT = """
너는 한국어 대화 요약 전문가이다.
입력된 문장은 'KoBART 요약(A)' 결과이며, 이를 가장 간결하고 정확한 한 문장 요약으로 변환해야 한다.

규칙:
- 반드시 **한 문장**으로만 요약할 것
- #Person1#, #Person2#, #Person3# 태그는 절대 삭제하거나 변형하지 말 것
- <usr>, <s>, </s>, [요약], "핵심", "요약:" 등의 접두사 생성 금지
- 새로운 정보나 추측(상상) 금지
- 대화를 기반으로 가장 핵심적인 사실만 남길 것
"""


# ---------------------------
#   Solar API 호출 함수
# ---------------------------
def call_solar(user_input: str):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "solar-pro",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    for attempt in range(3):
        try:
            if DEBUG_MODE:
                print("\n📌 --- Solar 요청 Payload ---")
                print(json.dumps(payload, indent=2, ensure_ascii=False))

            response = requests.post(API_URL, headers=headers, json=payload)

            if DEBUG_MODE:
                print("\n📌 --- Solar Raw 응답 ---")
                print("Status:", response.status_code)
                print(response.text[:500])

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[Solar API] 실패. Attempt {attempt+1} - Code: {response.status_code}")
                time.sleep(1)  # 재시도

        except Exception as e:
            print(f"[Solar API] 오류 발생: {e}")
            time.sleep(1)

    return ""  # 3회 실패 시 공백 반환


# ---------------------------
#     요약 생성 실행부
# ---------------------------
def run_solar_inference():
    print("📌 Solar Summarization 시작\n")

    df = pd.read_csv(INPUT_CSV)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row["fname"]
        summary_a = str(row["summary"])

        if summary_a.strip() == "":
            results.append([fname, ""])
            continue

        # Solar에 전달할 user prompt
        user_prompt = f"다음 요약문을 한 문장으로 재요약해줘:\n{summary_a}"

        summary_b = call_solar(user_prompt)
        results.append([fname, summary_b])

    out_df = pd.DataFrame(results, columns=["fname", "summary"])
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n🎉 Solar 요약 완료 → {OUTPUT_CSV}")


if __name__ == "__main__":
    run_solar_inference()
