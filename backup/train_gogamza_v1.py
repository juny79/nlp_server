from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import re
import pandas as pd
import wandb
import os

# wandb 초기화 (환경 변수나 직접 로그인 필요)
# 터미널에서: wandb login
# 또는 API key 설정: os.environ["WANDB_API_KEY"] = "your_api_key"

# 1) 선택 모델: KoBART (한국어 요약에 특화)
MODEL_NAME = "gogamza/kobart-summarization"  # ⭐ 한국어 대화 요약 모델

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2) 간단 전처리 함수
def clean_dialogue(text):
    text = re.sub(r'https?://\S+', '', text)  # URL 제거
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', text)  # ㅋㅋ, ㅎㅎ 제거
    text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)  # 이모지 제거
    text = re.sub(r' +', ' ', text)
    return text.strip()

# 3) Dataset 정의
class DialogSumDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=512, target_len=120):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_len = target_len

    def __getitem__(self, idx):
        dialog = clean_dialogue(self.df["dialogue"][idx])
        summary = self.df["summary"][idx]

        model_input = f"<SYS> {dialog} <SUMMARY>"
        enc = tokenizer(
            model_input,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        dec = tokenizer(
            summary,
            max_length=self.target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": dec["input_ids"].squeeze()
        }

    def __len__(self):
        return len(self.df)

# 4) 데이터 로딩
train_df = pd.read_csv("./data/train.csv")
eval_df = pd.read_csv("./data/dev.csv")

train_dataset = DialogSumDataset(train_df, tokenizer)
eval_dataset = DialogSumDataset(eval_df, tokenizer)

# 5) Model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 6) wandb 초기화
wandb.init(
    project="dialogsum_solar",
    name="solar_v1_finetune",
    tags=["solar-mini", "dialogue-summarization", "korean"],
    notes="Solar-mini-instruct fine-tuning for Korean dialogue summarization",
    config={
        "model": MODEL_NAME,
        "learning_rate": 3e-5,
        "epochs": 10,
        "batch_size": 16,
        "warmup_ratio": 0.2,
    }
)

# 7) Training arguments (최적화 버전)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_ratio=0.2,
    num_train_epochs=10,
    per_device_train_batch_size=8,  # GPU 메모리 고려
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch = 32
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=120,
    fp16=True,
    load_best_model_at_end=True,
    save_total_limit=3,
    report_to="wandb",  # wandb 로깅 활성화
    run_name="solar_v1_finetune"
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# 학습 완료 후 wandb 종료
wandb.finish()
