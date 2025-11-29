# train.py
import pandas as pd
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataset import SummDataset
from model import load_kobart_with_tokens

def train():
    train_df = pd.read_csv("train_prepared.csv")
    dev_df = pd.read_csv("dev_prepared.csv")

    SPECIAL_TOKENS = ["#Person1#", "#Person2#", "#Person3#", "#Address#", "#PhoneNumber#"]
    tokenizer, model = load_kobart_with_tokens(SPECIAL_TOKENS)

    train_dataset = SummDataset(train_df, tokenizer)
    dev_dataset = SummDataset(dev_df, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        weight_decay=0.01,
        warmup_steps=300,
        fp16=False,  # CPU/AMD 환경 고려
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    tokenizer.save_pretrained("./checkpoints/best_model")
    model.save_pretrained("./checkpoints/best_model")

    print("✓ 학습 완료 및 모델 저장")

if __name__ == "__main__":
    train()
