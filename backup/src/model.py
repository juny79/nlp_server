# model.py
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def load_kobart_with_tokens(new_tokens):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")

    # special tokens 추가
    tokenizer.add_tokens(new_tokens)

    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
