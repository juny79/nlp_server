import pandas as pd
import torch
from torch.utils.data import Dataset

class DialogDataset(Dataset):
    def __init__(self, df, tokenizer, encoder_max_len, decoder_max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dialogue = self.df.iloc[idx]["dialogue"]
        summary = self.df.iloc[idx]["summary"]

        # Encoder
        enc = self.tokenizer(
            dialogue,
            max_length=self.encoder_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Decoder
        dec = self.tokenizer(
            summary,
            max_length=self.decoder_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = dec["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
