import pandas as pd
import os
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration

class Preprocess:
    def __init__(self, bos, eos):
        self.bos = bos
        self.eos = eos

    def make_set_as_df(self, file_path):
        df = pd.read_csv(file_path)
        return df[['fname', 'dialogue']]

    def make_input(self, df):
        enc = df['dialogue']
        dec = [self.bos] * len(df)
        return enc.tolist(), dec


class DatasetForInference(Dataset):
    def __init__(self, encoder_input, ids):
        self.encoder_input = encoder_input
        self.ids = ids
        self.length = len(ids)

    def __getitem__(self, idx):
        item = {k: v[idx].clone() for k, v in self.encoder_input.items()}
        item['ID'] = self.ids[idx]
        return item

    def __len__(self):
        return self.length


def load_model_for_inference(config, device):
    model_name = config['general']['model_name']
    checkpoint_path = config['inference']['ckt_path']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": config['tokenizer']['special_tokens']})

    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return model, tokenizer


def inference():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_for_inference(config, device)

    preproc = Preprocess(config['tokenizer']['bos_token'],
                         config['tokenizer']['eos_token'])

    test_df = preproc.make_set_as_df(os.path.join(config['general']['data_path'], "test.csv"))
    enc, dec = preproc.make_input(test_df)

    tok_enc = tokenizer(enc, return_tensors="pt", padding=True, truncation=True,
                        max_length=config['tokenizer']['encoder_max_len'])

    ds = DatasetForInference(tok_enc, test_df['fname'].tolist())
    loader = DataLoader(ds, batch_size=config['inference']['batch_size'])

    outputs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            gen_ids = model.generate(
                input_ids=batch['input_ids'].to(device),
                max_length=config['inference']['generate_max_length'],
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            for ids in gen_ids:
                outputs.append(tokenizer.decode(ids, skip_special_tokens=True))

    # Remove noise tokens
    clean = outputs.copy()
    for t in config['inference']['remove_tokens']:
        clean = [c.replace(t, " ") for c in clean]

    df_out = pd.DataFrame({"fname": test_df['fname'], "summary": clean})
    os.makedirs(config['inference']['result_path'], exist_ok=True)
    df_out.to_csv(os.path.join(config['inference']['result_path'], "output.csv"), index=False)

    print("Inference Done → output.csv 생성 완료")


if __name__ == "__main__":
    inference()
