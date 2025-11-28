import re

def clean_dialogue(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)   # URL 제거
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)   # ㅋㅋ,ㅎㅎ 제거
    text = re.sub(r"[^\S\n]+", " ", text)      # 공백 normalize
    text = re.sub(r"[\u2600-\u27BF]", "", text)  # 이모지 삭제
    return text.strip()

def build_prompt(dialogue: str, template: str) -> str:
    dialogue = clean_dialogue(dialogue)
    return template.replace("{dialogue}", dialogue)
