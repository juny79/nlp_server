from .preprocess import build_prompt

def make_prompt(dialogue: str, template: str) -> str:
    return build_prompt(dialogue, template)
