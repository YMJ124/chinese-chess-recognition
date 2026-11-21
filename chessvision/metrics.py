from __future__ import annotations

import numpy as np


def classification_metrics(confusion: np.ndarray) -> dict:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    f1_values: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    for index in range(confusion.shape[0]):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "total": total,
        "correct": correct,
    }
