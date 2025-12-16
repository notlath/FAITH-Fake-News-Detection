import os
import torch
import numpy as np
import evaluate
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from dataset import load_isot_dataset

# Configuration
MODEL_CHECKPOINT = "distilroberta-base"  # Modern, efficient Transformer
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "models" / "trained_model"
LOG_DIR = BASE_DIR / "models" / "logs"


def compute_metrics(eval_pred):
    """Computes accuracy and f1 score."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)

    return {**accuracy, **f1}


def main():
    print(f"Initializing training for {MODEL_CHECKPOINT}...")

    # 1. Load Data
    try:
        dataset = load_isot_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # We will pad dynamically with DataCollator
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Trainer Setup
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(LOG_DIR),
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("Starting training...")
    trainer.train()

    # 6. Save Final Model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Training complete!")


if __name__ == "__main__":
    main()
