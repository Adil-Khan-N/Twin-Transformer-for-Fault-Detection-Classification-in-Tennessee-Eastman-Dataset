import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    if len(y_true) == 0:
        raise ValueError("No samples were found in the provided dataloader.")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    num_classes = cm.shape[0]
    labels = list(range(num_classes))
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_accuracy = np.zeros(num_classes, dtype=np.float32)
    row_sums = cm.sum(axis=1).astype(np.float32)
    nonzero_mask = row_sums != 0
    per_class_accuracy[nonzero_mask] = cm.diagonal()[nonzero_mask] / row_sums[nonzero_mask]

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "per_class": {
            "accuracy": per_class_accuracy,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
        },
    }
