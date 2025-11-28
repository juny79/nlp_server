# placeholder for src/preprocess.py
import re
import pandas as pd

class DialoguePreprocessor:
    def __init__(self, bos, eos):
        self.bos = bos
        self.eos = eos

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        remove = ["음", "에", "아", "어", "그", "그러니까", "그래서", "그니까"]
        for r in remove:
            text = text.replace(r, "")
        return text.strip()

    def parse_speakers(self, dialogue):
        lines = dialogue.split("\n")
        out = []
        for line in lines:
            if "#Person" in line:
                out.append(line)
        return "\n".join(out)

    def format_for_encoder(self, dialogue):
        dialog_clean = self.clean_text(dialogue)
        speakers = self.parse_speakers(dialog_clean)

        return (
            "[대화요약]\n"
            f"{speakers}\n"
            "[내용]\n"
            f"{dialog_clean}"
        )

    def prepare_train(self, df):
        df["dialogue"] = df["dialogue"].apply(self.format_for_encoder)
        return df
