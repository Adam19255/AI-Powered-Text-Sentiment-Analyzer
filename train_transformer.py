"""
Fine-tune DistilBERT on the IMDB dataset (local aclImdb files).
Saves the fine-tuned model & tokenizer under ./models/distilbert-imdb/
"""

from pprint import pprint

# core libs
import numpy as np

# huggingface libs
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
from sklearn.utils import shuffle

# sklearn metrics
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# load the loader you already have
from inspect_data import load_imdb_dataset

def compute_metrics(p):
    """Compute accuracy and F1 (macro) for Trainer.evaluate/predict."""
    preds = p.predictions
    if isinstance(preds, tuple):  # some models return tuple (logits, hidden_states)
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = p.label_ids

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1_per_label, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
    return {"accuracy": acc, "f1_macro": f1, "precision_per_label": precision.tolist(), "recall_per_label": recall.tolist()}

def build_hf_dataset(train_texts, train_labels, test_texts, test_labels, val_fraction=0.10, random_seed=42):
    """
    Convert python lists into HuggingFace DatasetDict with train/validation/test splits.
    val_fraction: fraction of training set to reserve for validation (default 0.10)
    """
    # Shuffle training set first (so validation sample is balanced)
    train_texts, train_labels = shuffle(train_texts, train_labels, random_state=random_seed)

    # Use a small validation split off the training set
    n_train_total = len(train_texts)
    n_val = int(n_train_total * val_fraction)

    # validation = last n_val, train = remaining
    val_texts = train_texts[:n_val]
    val_labels = train_labels[:n_val]
    new_train_texts = train_texts[n_val:]
    new_train_labels = train_labels[n_val:]

    # Prepare HF datasets (they expect dict with "text" and "label")
    train_ds = Dataset.from_dict({"text": new_train_texts, "label": new_train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

def tokenize_batch(examples, tokenizer, max_length=256):
    # Tokenize a batch of texts
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def main(
    model_name="distilbert-base-uncased",
    output_dir="models/distilbert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    val_fraction=0.10,
    max_length=256,
    random_seed=42,
    fp16_if_available=True,
):
    # 1) LOAD raw data (your loader returns lists)
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset()
    print(f"Loaded train: {len(train_texts)}, test: {len(test_texts)}")

    # 2) Build HF DatasetDict with small validation split
    ds = build_hf_dataset(train_texts, train_labels, test_texts, test_labels, val_fraction=val_fraction, random_seed=random_seed)
    print(ds)

    # 3) Load tokenizer & model
    print("Loading tokenizer and model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Tokenize datasets (batched)
    print("Tokenizing datasets... (this may take a minute)")
    tokenized = ds.map(lambda examples: tokenize_batch(examples, tokenizer, max_length=max_length), batched=True)

    # 5) Set format to PyTorch tensors for Trainer
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    # 6) Training arguments
    # Choose fp16 if GPU is available and user allowed it
    import torch
    use_fp16 = fp16_if_available and torch.cuda.is_available()
    print("CUDA available:", torch.cuda.is_available(), "Using fp16:", use_fp16)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        seed=random_seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # 7) Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8) Train
    print("Starting training — this may take time. Check GPU usage if available.")
    trainer.train()

    # 9) Evaluate on test set
    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized["test"])
    pprint(results)

    # 10) Save model & tokenizer
    print("Saving final model to:", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done. Model & tokenizer saved.")

if __name__ == "__main__":
    # you can also expose command-line args later; for now run main() with defaults
    main()
