# placeholder for src/model.py
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

def load_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["general"]["model_name"])
    tokenizer.add_special_tokens({
        "additional_special_tokens": cfg["tokenizer"]["special_tokens"]
    })

    bart_cfg = BartConfig.from_pretrained(cfg["general"]["model_name"])
    bart_cfg.no_repeat_ngram_size = cfg["inference"]["no_repeat_ngram_size"]
    bart_cfg.forced_eos_token_id = tokenizer.eos_token_id

    model = BartForConditionalGeneration.from_pretrained(cfg["general"]["model_name"], config=bart_cfg)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
