# placeholder for train.py
import pandas as pd
import torch
from src.utils import load_config
from src.preprocess import DialoguePreprocessor
from src.dataset import SummDataset
from src.model import load_model
from src.trainer import build_trainer

def main():
    cfg = load_config("./config/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    train_df = pd.read_csv(cfg["general"]["data_path"] + "train.csv")
    val_df = pd.read_csv(cfg["general"]["data_path"] + "dev.csv")

    pre = DialoguePreprocessor(
        cfg["tokenizer"]["bos_token"],
        cfg["tokenizer"]["eos_token"]
    )

    train_df = pre.prepare_train(train_df)
    val_df = pre.prepare_train(val_df)

    tokenizer, model = load_model(cfg)
    model.to(device)

    # Tokenize training data
    encoder_input_train = train_df["dialogue"].tolist()
    decoder_input_train = (cfg["tokenizer"]["bos_token"] + train_df["summary"]).tolist()
    decoder_output_train = (train_df["summary"] + cfg["tokenizer"]["eos_token"]).tolist()
    
    tokenized_encoder_train = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, 
                            max_length=cfg["tokenizer"]["encoder_max_len"], return_token_type_ids=False)
    tokenized_decoder_in_train = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, 
                        max_length=cfg["tokenizer"]["decoder_max_len"], return_token_type_ids=False)
    tokenized_decoder_out_train = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, 
                        max_length=cfg["tokenizer"]["decoder_max_len"], return_token_type_ids=False)

    # Tokenize validation data
    encoder_input_val = val_df["dialogue"].tolist()
    decoder_input_val = (cfg["tokenizer"]["bos_token"] + val_df["summary"]).tolist()
    decoder_output_val = (val_df["summary"] + cfg["tokenizer"]["eos_token"]).tolist()
    
    tokenized_encoder_val = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, 
                            max_length=cfg["tokenizer"]["encoder_max_len"], return_token_type_ids=False)
    tokenized_decoder_in_val = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, 
                        max_length=cfg["tokenizer"]["decoder_max_len"], return_token_type_ids=False)
    tokenized_decoder_out_val = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, 
                        max_length=cfg["tokenizer"]["decoder_max_len"], return_token_type_ids=False)

    train_set = SummDataset(tokenized_encoder_train, tokenized_decoder_in_train, 
                            tokenized_decoder_out_train, len(encoder_input_train))
    val_set = SummDataset(tokenized_encoder_val, tokenized_decoder_in_val, 
                          tokenized_decoder_out_val, len(encoder_input_val))

    trainer = build_trainer(cfg, model, tokenizer, train_set, val_set)
    trainer.train()
    trainer.save_model(cfg["general"]["save_dir"])

if __name__ == "__main__":
    main()
