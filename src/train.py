from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


@dataclass
class TextDataset(Dataset):
    texts: list
    labels: list
    tokenizer: AutoTokenizer
    max_length: int = 128

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_model(model_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def tokenize_dataset(tokenizer, texts, labels) -> TextDataset:
    return TextDataset(texts=texts, labels=labels, tokenizer=tokenizer)


def train_model(
    model_name: str,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    output_dir: str = "./model",
) -> Tuple[Trainer, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_model(model_name, num_labels=len(set(train_labels)))

    train_dataset = tokenize_dataset(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_dataset(tokenizer, val_texts, val_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return trainer, tokenizer
