import os
import pandas as pd
from torch.utils.data import Dataset

from model import load_kobart_model
from trainer import get_trainer


############################################################
# Dataset 클래스
############################################################

class DialogueSummaryDataset(Dataset):

    def __init__(self, df, tokenizer, max_input=640, max_output=100):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        dialog = row["dialogue"]
        summary = row["summary"]

        model_inputs = self.tokenizer(
            dialog,
            truncation=True,
            padding="max_length",
            max_length=self.max_input,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            summary,
            truncation=True,
            padding="max_length",
            max_length=self.max_output,
            return_tensors="pt"
        )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels

        return {key: val.squeeze() for key, val in model_inputs.items()}


############################################################
# 데이터 로딩 함수
############################################################

def load_dataset(data_path="./data/"):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    dev_df = pd.read_csv(os.path.join(data_path, "dev.csv"))
    return train_df, dev_df


############################################################
# main 학습 실행
############################################################

def main():

    # 1) KoBART Model + Tokenizer 로드
    tokenizer, model = load_kobart_model(
        model_name="digit82/kobart-summarization"
    )

    # 2) 데이터 불러오기
    train_df, dev_df = load_dataset()

    # 3) Dataset 생성
    train_dataset = DialogueSummaryDataset(train_df, tokenizer)
    eval_dataset = DialogueSummaryDataset(dev_df, tokenizer)

    # 4) Trainer 생성
    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset)

    # 5) 학습 실행
    trainer.train()

    # 6) 최종 저장
    trainer.save_model("./checkpoints/best_model/")
    tokenizer.save_pretrained("./checkpoints/best_model/")

    print("학습 완료")


if __name__ == "__main__":
    main()
