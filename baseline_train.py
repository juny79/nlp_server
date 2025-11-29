import pandas as pd
import os
import yaml
import torch
from glob import glob
from pprint import pprint
from tqdm import tqdm
from rouge import Rouge
import wandb

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)

# ------------------------------------------------------------
# Custom callback: 학습 중 손실/평가 출력 강화를 위한 Callback
# ------------------------------------------------------------
class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"📝 LOG | Epoch: {state.epoch:.2f}, Step: {state.global_step}, Logs: {logs}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print("\n📊 ===== Evaluation Metrics =====")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("================================\n")


# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str):
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_in = dataset['dialogue']
            decoder_in = [self.bos_token] * len(dataset)
            return encoder_in.tolist(), decoder_in
        else:
            encoder_in = dataset['dialogue']
            decoder_in = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_out = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_in.tolist(), decoder_in.tolist(), decoder_out.tolist()


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {k: v[idx].clone() for k, v in self.encoder_input.items()}
        item_dec = {k: v[idx].clone() for k, v in self.decoder_input.items()}

        item_dec['decoder_input_ids'] = item_dec.pop('input_ids')
        item_dec['decoder_attention_mask'] = item_dec.pop('attention_mask')

        item.update(item_dec)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.length


class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {k: v[idx].clone() for k, v in self.encoder_input.items()}
        item_dec = {k: v[idx].clone() for k, v in self.decoder_input.items()}

        item_dec['decoder_input_ids'] = item_dec.pop('input_ids')
        item_dec['decoder_attention_mask'] = item_dec.pop('attention_mask')

        item.update(item_dec)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.length


# ------------------------------------------------------------
# Build dataset
# ------------------------------------------------------------
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_df = preprocessor.make_set_as_df(os.path.join(data_path, "train.csv"))
    val_df = preprocessor.make_set_as_df(os.path.join(data_path, "dev.csv"))

    enc_tr, dec_tr, out_tr = preprocessor.make_input(train_df)
    enc_val, dec_val, out_val = preprocessor.make_input(val_df)

    tok_enc_tr = tokenizer(enc_tr, return_tensors="pt", padding=True, truncation=True,
                           max_length=config['tokenizer']['encoder_max_len'])
    tok_dec_tr = tokenizer(dec_tr, return_tensors="pt", padding=True, truncation=True,
                           max_length=config['tokenizer']['decoder_max_len'])
    tok_out_tr = tokenizer(out_tr, return_tensors="pt", padding=True, truncation=True,
                           max_length=config['tokenizer']['decoder_max_len'])

    tok_enc_val = tokenizer(enc_val, return_tensors="pt", padding=True, truncation=True,
                            max_length=config['tokenizer']['encoder_max_len'])
    tok_dec_val = tokenizer(dec_val, return_tensors="pt", padding=True, truncation=True,
                            max_length=config['tokenizer']['decoder_max_len'])
    tok_out_val = tokenizer(out_val, return_tensors="pt", padding=True, truncation=True,
                            max_length=config['tokenizer']['decoder_max_len'])

    train_ds = DatasetForTrain(tok_enc_tr, tok_dec_tr, tok_out_tr, len(enc_tr))
    val_ds = DatasetForVal(tok_enc_val, tok_dec_val, tok_out_val, len(enc_val))

    print(f"🔹 Train dataset: {len(train_ds)} samples")
    print(f"🔹 Val dataset: {len(val_ds)} samples")

    return train_ds, val_ds


# ------------------------------------------------------------
# ROUGE metric
# ------------------------------------------------------------
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()

    predictions = pred.predictions
    labels = pred.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    remove_tokens = config["inference"]["remove_tokens"]
    clean_preds = decoded_preds.copy()
    clean_labels = decoded_labels.copy()

    for t in remove_tokens:
        clean_preds = [p.replace(t, " ") for p in clean_preds]
        clean_labels = [l.replace(t, " ") for l in clean_labels]

    scores = rouge.get_scores(clean_preds, clean_labels, avg=True)

    return {k: v["f"] for k, v in scores.items()}


# ------------------------------------------------------------
# Load model/tokenizer
# ------------------------------------------------------------
def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens({"additional_special_tokens": config['tokenizer']['special_tokens']})

    bart_config = BartConfig.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return model, tokenizer


# ------------------------------------------------------------
# Training main
# ------------------------------------------------------------
def main():
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("[DEBUG] Loaded config:")
        pprint(config)
        if config is None:
            raise ValueError("YAML parsing returned None. Check config/config.yaml for syntax errors or empty file.")
    except Exception as e:
        print(f"[ERROR] Failed to load config/config.yaml: {e}")
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training start | Device = {device}")

    # Load model
    model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # Build dataset
    preproc = Preprocess(config["tokenizer"]["bos_token"],
                         config["tokenizer"]["eos_token"])
    train_ds, val_ds = prepare_train_dataset(config, preproc,
                                             config["general"]["data_path"],
                                             tokenizer)

    # Training args (터미널 로그 강화)
    args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        logging_strategy="steps",
        log_level="info",
        disable_tqdm=False,        # tqdm progress bar ON
        fp16=False,
        predict_with_generate=True,
        generation_max_length=config['training']['generation_max_length'],
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda x: compute_metrics(config, tokenizer, x),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            PrintMetricsCallback()   # 🔥 터미널 출력 강화
        ]
    )

    print("📌 Trainer initialized. Starting training...\n")
    trainer.train()
    trainer.save_model("./checkpoints/best_model")

    print("\n🎉 Training complete! Best model saved at ./checkpoints/best_model")


if __name__ == "__main__":
    main()
