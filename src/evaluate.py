from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import Trainer


def evaluate(trainer: Trainer, labels: List[int]):
    predictions = trainer.predict(trainer.eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    report = classification_report(labels, preds)
    return acc, f1, report
