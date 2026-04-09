import copy

import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import numpy as np


class EvolveGCN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.linear(x).squeeze()


def compute_class_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return torch.tensor(neg / pos, dtype=torch.float)


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


def run_evolve_gcn(df_train, df_val, df_test, edges):
    print("\n===== EVOLVE GCN MODEL =====")

    feature_cols = [col for col in df_train.columns if col not in ["txId", "class"]]

    x_train = torch.tensor(df_train[feature_cols].values, dtype=torch.float)
    y_train = torch.tensor(df_train["class"].values, dtype=torch.float)

    x_val = torch.tensor(df_val[feature_cols].values, dtype=torch.float)
    y_val = torch.tensor(df_val["class"].values, dtype=torch.float)

    x_test = torch.tensor(df_test[feature_cols].values, dtype=torch.float)
    y_test = torch.tensor(df_test["class"].values, dtype=torch.float)

    model = EvolveGCN(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pos_weight = compute_class_weight(y_train)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1
    best_state = copy.deepcopy(model.state_dict())
    patience = 10
    counter = 0

    for epoch in range(1, 201):
        model.train()
        logits = model(x_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val)).numpy()
            val_pr = average_precision_score(y_val.numpy(), val_probs)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss {loss:.4f} | Val PR-AUC {val_pr:.4f}")

        if val_pr > best_val:
            best_val = val_pr
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping!")
            break

    model.load_state_dict(best_state)

    with torch.no_grad():
        test_probs = torch.sigmoid(model(x_test)).numpy()

    best_t = find_best_threshold(y_val.numpy(), val_probs)
    y_pred = (test_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(y_test.numpy(), test_probs),
        "F1": f1_score(y_test.numpy(), y_pred),
        "Precision": precision_score(y_test.numpy(), y_pred),
        "Recall": recall_score(y_test.numpy(), y_pred),
        "model": "EvolveGCN"
    }

    print("\n===== RESULTADOS EVOLVE GCN =====")
    print(results)

    return results
