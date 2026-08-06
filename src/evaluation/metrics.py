import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_timestep(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    pr_auc = (
        average_precision_score(y_true, y_prob)
        if len(np.unique(y_true)) >= 2
        else np.nan
    )

    return {
        "PR_AUC": pr_auc,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "n_pos": int((y_true == 1).sum()),
        "n_total": len(y_true),
    }
