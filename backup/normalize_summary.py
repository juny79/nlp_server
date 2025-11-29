import re

############################################################
# 1) 태그 유지 및 복원 시스템
############################################################

def wrap_tags(text: str):
    """#Person1# → <P1> 형태로 변환 (내부용 보호)"""
    def repl(m):
        return f"<P{m.group(1)}>"
    return re.sub(r"#Person(\d+)#", repl, text)


def unwrap_tags(text: str):
    """<P1> → #Person1# 최종 제출용"""
    def repl(m):
        return f"#Person{m.group(1)}#"
    return re.sub(r"<P(\d+)>", repl, text)


############################################################
# 2) 문장 정리 / 문장 분리기
############################################################

def split_sentences(text: str):
    """문장을 . ? ! 기준으로 분리하는 함수"""
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


############################################################
# 3) 핵심 문장 선택기
############################################################

def pick_main_sentences(sentences):
    """
    요약에 알맞은 문장 1~2개를 선택.
    우선순위:
    1) 태그 포함 문장
    2) 내용 밀도 높은 문장 (10자 이상)
    3) 첫 문장
    """
    if not sentences:
        return []

    tagged = [s for s in sentences if "<P" in s]
    if len(tagged) >= 1:
        # 태그 포함 문장 우선 1~2개
        return tagged[:2]

    # 태그 비포함 시 — 내용 길이 기반
    long_sents = [s for s in sentences if len(s) > 10]
    if long_sents:
        return long_sents[:2]

    # 최후 — 첫 문장만
    return sentences[:1]


############################################################
# 4) 두 문장 → 한 문장 자연 결합기
############################################################

def merge_two_sentences(sent_list):
    """
    두 문장을 자연스럽게 '그리고' 기반으로 결합.
    """
    if not sent_list:
        return ""

    if len(sent_list) == 1:
        return sent_list[0]

    a, b = sent_list[0], sent_list[1]

    # 결합 규칙
    merged = f"{a} 그리고 {b}"
    return merged


############################################################
# 5) 스타일 통일기 (정답 summary 스타일로 변환)
############################################################

def enforce_label_style(text: str):
    """
    정답 summary 스타일:
    - 인물 태그(#Person1#) 그대로 유지
    - “~한다” 또는 “~했다” 형태로 통일
    - 불필요한 서술/감탄/메타표현 제거
    """
    s = text

    # Markdown 제거
    s = re.sub(r"[*_`>-]+", "", s)

    # 영어 존칭 제거
    s = re.sub(r"\b(Mr|Ms|Mrs|Dr)\.?\s+", "", s)

    # 메타 표현 제거
    bad = [
        r"요약하면", r"핵심은", r"간단히 말해", r"결국", r"즉",
        r"요약:", r"Summary", r"핵심 요약", r"한 문장 요약",
        r"다음은", r"다음 내용은"
    ]
    for b in bad:
        s = re.sub(b, "", s, flags=re.IGNORECASE)

    # 반복 공백 제거
    s = re.sub(r"\s+", " ", s).strip()

    # 한국어 시제 정규화 (했어요 → 했다 / 합니다 → 한다)
    s = re.sub(r"했어요", "했다", s)
    s = re.sub(r"합니다", "한다", s)
    s = re.sub(r"합니다\.", "한다.", s)
    s = re.sub(r"했다\.", "했다.", s)

    # ending 정리
    if not s.endswith("."):
        s += "."

    return s


############################################################
# 6) 전체 normalize 처리
############################################################

def normalize_summary(text: str):
    """
    Solar/BART/meta-ranker 결과를 한 문장 summary로 고도화하는 최종 함수.
    """

    if text is None or str(text).strip() == "":
        return ""

    # 내부 보호 태그 적용
    wrapped = wrap_tags(text)

    # 문장 분리
    sentences = split_sentences(wrapped)

    # 핵심 문장 선택
    chosen = pick_main_sentences(sentences)

    # 결합
    merged = merge_two_sentences(chosen)

    # 스타일 강제 변환
    styled = enforce_label_style(merged)

    # 태그 복원
    final_text = unwrap_tags(styled)

    return final_text.strip()


############################################################
# 7) 파일 전체 후처리용 함수
############################################################

def normalize_file(input_csv, output_csv):
    """
    summary 컬럼에 normalize_summary 적용 후 저장.
    """
    import pandas as pd
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[에러] 입력 파일 읽기 실패: {input_csv}")
        print(f"[상세] {e}")
        return
    try:
        df["summary"] = df["summary"].apply(normalize_summary)
        df.to_csv(output_csv, index=False)
        print(f"[완료] 정규화된 파일 저장 → {output_csv}")
    except Exception as e:
        print(f"[에러] 파일 저장 또는 처리 실패: {output_csv}")
        print(f"[상세] {e}")


############################################################
# 8) CLI 실행
############################################################

if __name__ == "__main__":
    # 예: meta_ranker 결과를 읽어서 normalize
    normalize_file("final_ensemble_output.csv", "summary_final.csv")
