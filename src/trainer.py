# placeholder for src/trainer.py
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

def build_trainer(cfg, model, tokenizer, train_set, val_set):

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["general"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        fp16=cfg["training"]["fp16"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        learning_rate=cfg["training"]["learning_rate"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        predict_with_generate=cfg["training"]["predict_with_generate"],
        logging_steps=50,
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        seed=cfg["training"]["seed"],
        logging_dir=cfg["training"]["logging_dir"],
        logging_strategy=cfg["training"]["logging_strategy"],
        generation_max_length=cfg["training"]["generation_max_length"],
        do_train=cfg["training"]["do_train"],
        do_eval=cfg["training"]["do_eval"],
        report_to=cfg["training"]["report_to"],
        optim=cfg["training"]["optim"]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                early_stopping_threshold=cfg["training"]["early_stopping_threshold"]
            )
        ]
    )
    return trainer
