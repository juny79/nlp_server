# dataset.py
import torch
from torch.utils.data import Dataset

class SummDataset(Dataset):
    def __init__(self, df, tokenizer, max_input=640, max_output=96):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dialogue = row["dialogue"]
        summary = row["summary"]

        x = self.tokenizer(
            dialogue,
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        y = self.tokenizer(
            summary,
            max_length=self.max_output,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = y["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": x["input_ids"].squeeze(),
            "attention_mask": x["attention_mask"].squeeze(),
            "labels": labels
        }
