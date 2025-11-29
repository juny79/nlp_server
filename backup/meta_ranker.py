# meta_ranker.py
import re
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------
# 1) 라벨 스타일 정규화 (공통 후처리)
# ---------------------------------------------------------
def normalize_summary(s: str) -> str:
    if not s or len(s.strip()) == 0:
        return ""

    # Markdown 제거
    s = re.sub(r"[*_`#>-]+", "", s)

    # 제목 패턴 제거
    s = re.sub(r"(요약|핵심|정리|간결 버전).*?:", "", s)

    # 여러 문장 → 첫 문장만
    sent = re.split(r"[.!?]", s)
    sent = [x.strip() for x in sent if len(x.strip()) > 5]

    if len(sent) == 0:
        return ""

    s = sent[0]

    # 연속 공백 제거
    s = re.sub(r"\s+", " ", s).strip()

    # 종결형 통일
    if not s.endswith("다."):
        s = s.rstrip(" .") + "다."

    return s


# ---------------------------------------------------------
# 2) 정답 패턴 기반 점수 (대회용 정규화)
# ---------------------------------------------------------
def pattern_score(summary: str) -> float:
    score = 0.0

    # #PersonX# 등장 → 대회 summary와 유사
    if re.search(r"#Person\d+#", summary):
        score += 1.0

    # 한 문장 여부
    if len(re.split(r"[.!?]", summary)) <= 2:
        score += 1.0

    # 현재형 '다' 종결 체크
    if summary.endswith("다.") or summary.endswith("다"):
        score += 1.0

    # 불필요한 연결어가 없으면 가산
    if not summary.startswith(("그리고", "그러나", "하지만", "또한")):
        score += 1.0

    return score


# ---------------------------------------------------------
# 3) 대화 overlap 기반 점수
# ---------------------------------------------------------
def overlap_score(dialogue: str, summary: str) -> float:
    # dialogue 핵심 명사 후보 추출
    tokens = re.findall(r"[A-Za-z가-힣#]{2,}", dialogue)
    unique = list(set(tokens))

    hit = 0
    for t in unique:
        if t in summary:
            hit += 1

    # 정규화
    if len(unique) == 0:
        return 0.0

    return hit / len(unique)


# ---------------------------------------------------------
# 4) 길이 기반 점수 (정답 길이와 유사할수록 가산)
# ---------------------------------------------------------
def length_score(summary: str) -> float:
    length = len(summary)

    if 30 <= length <= 120:
        return 1.0
    if 120 < length <= 200:
        return 0.5
    if length < 20:
        return 0.2
    return 0.1


# ---------------------------------------------------------
# 5) 전체 스코어 계산
# ---------------------------------------------------------
def total_score(dialogue: str, summary: str) -> float:

    if len(summary.strip()) == 0:
        return 0.0

    return (
        pattern_score(summary) * 1.5 +     # 정답 패턴 반영 비중 큼
        overlap_score(dialogue, summary) * 1.0 +
        length_score(summary) * 1.0
    )


# ---------------------------------------------------------
# 6) Meta-ranker 앙상블
# ---------------------------------------------------------
def meta_ranker(test_path, summaryA_path, summaryB_path):

    df_test = pd.read_csv(test_path)
    df_A = pd.read_csv(summaryA_path)
    df_B = pd.read_csv(summaryB_path)

    df_A.set_index("fname", inplace=True)
    df_B.set_index("fname", inplace=True)

    outputs = []

    for i in tqdm(range(len(df_test)), desc="Meta-Ranking", dynamic_ncols=True, ascii=True):

        row = df_test.iloc[i]
        fname = row["fname"]
        dialogue = row["dialogue"]

        summary_A = normalize_summary(df_A.loc[fname]["summary"])
        summary_B = normalize_summary(df_B.loc[fname]["summary"])

        score_A = total_score(dialogue, summary_A)
        score_B = total_score(dialogue, summary_B)

        # 더 높은 점수를 최종 선택
        final_summary = summary_A if score_A >= score_B else summary_B

        outputs.append([fname, final_summary])

    out_df = pd.DataFrame(outputs, columns=["fname", "summary"])
    out_df.to_csv("final_ensemble_output.csv", index=False, encoding="utf-8-sig")
    print(">>> Saved to final_ensemble_output.csv")


if __name__ == "__main__":
    # 예시
    meta_ranker(
        test_path="./data/test.csv",
        summaryA_path="./summary_A_model_output.csv",
        summaryB_path="./summary_B_solar_output.csv"
    )
