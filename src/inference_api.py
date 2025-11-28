import os
import time
import re
import pandas as pd
from tqdm import tqdm
from .utils import load_api_key, call_solar_api
from .prompt import make_prompt


def extract_sentences(text):
    """Solar 출력에서 실제 후보 문장만 뽑아낸다."""
    if not text:
        return []

    # 따옴표 안 문장이 있으면 그걸 우선
    m = re.findall(r"“([^”]+)”|\"([^\"]+)\"", text)
    quotes = [q[0] or q[1] for q in m if (q[0] or q[1])]
    if quotes:
        return quotes

    # markdown/list 제거
    text = re.sub(r"\*\*.*?\*\*", "", text)
    text = re.sub(r"^\s*[-•*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text, flags=re.MULTILINE)

    # 괄호 내용 제거
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^]]*\]", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)

    # HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)

    # "요약:", "핵심:" 등 제거
    text = re.sub(r"(요약|핵심|정리|결론|main|summary)[:：]?", "", text, flags=re.IGNORECASE)

    # 문장 split
    candidates = re.split(r"[.!?]\s+", text)
    candidates = [c.strip() for c in candidates if len(c.strip()) > 5]

    return candidates


def compress_sentences(sentences):
    """후보 문장 목록에서 핵심 1~2문장만 남긴다."""
    if not sentences:
        return ""

    # 너무 긴 문장 삭제
    sentences = [s for s in sentences if len(s) < 200]

    # 의미없거나 패턴 문장 삭제
    ban_patterns = [
        r"^요약", r"^핵심", r"^정리", r"^결론",
        r"^다음", r"^본 대화", r"^이 대화",
    ]
    filtered = []
    for s in sentences:
        if not any(re.match(bp, s, re.IGNORECASE) for bp in ban_patterns):
            filtered.append(s)

    if not filtered:
        filtered = sentences

    # 앞쪽 문장이 일반적으로 요약에 적합 → 선두 1~2개 선택
    result = filtered[:2]

    # 최종 길이 제한
    final = " ".join(result)
    if len(final) > 160:
        final = final[:160]

    return final.strip()


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
        raw = call_solar_api(base_url, api_key, model, prompt)

        # Step1: 문장 후보 추출
        candidates = extract_sentences(raw)

        # Step2: 핵심 문장만 압축
        summary = compress_sentences(candidates)

        results.append({"fname": fname, "summary": summary})

        time.sleep(config["inference"]["sleep"])

    out = pd.DataFrame(results)
    os.makedirs(config["general"]["result_path"], exist_ok=True)
    out.to_csv(os.path.join(config["general"]["result_path"], "output.csv"), index=False)
    return out
