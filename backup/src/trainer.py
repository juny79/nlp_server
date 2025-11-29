from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def get_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir="./checkpoints"):

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        num_train_epochs=10,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # GPU 부족 대비
        warmup_ratio=0.1,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=100,
        fp16=True if model.device.type == "cuda" else False,
        report_to="none",
        seed=42
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    return trainer
