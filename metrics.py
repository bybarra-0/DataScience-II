import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from config import CLASS_NAMES

def compute_metrics(y_true, y_pred) -> dict:
    # Map integer predictions to canonical strings if passed as numbers
    if len(y_true) > 0 and isinstance(y_true[0], (int, np.integer)):
        y_true = [CLASS_NAMES[i] for i in y_true]
    if len(y_pred) > 0 and isinstance(y_pred[0], (int, np.integer)):
        y_pred = [CLASS_NAMES[i] for i in y_pred]

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    _, per_class_rec, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    return {
        "balanced_accuracy": round(float(bal_acc), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "per_class_recall": {
            cls: round(float(rec), 4) for cls, rec in zip(CLASS_NAMES, per_class_rec)
        },
        "confusion_matrix": cm.tolist()
    }