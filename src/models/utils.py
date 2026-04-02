import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 50)
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh
