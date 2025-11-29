# preprocess.py
import re

class DialoguePreprocessor:
    def __init__(self):
        pass

    def normalize_dialogue(self, text: str) -> str:
        # 줄바꿈 통일
        text = text.replace("\r", "\n")

        # 연속 개행 제거
        text = re.sub(r"\n+", "\n", text)

        # #PersonX#: → #PersonX#
        text = re.sub(r"(#Person\d+#):", r"\1", text)

        # 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_summary(self, text: str) -> str:
        # 공백 정리
        text = re.sub(r"\s+", " ", text).strip()
        return text
