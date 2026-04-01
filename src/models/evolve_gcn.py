import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

from src.models.gcn import build_pyg_data, evaluate
from src.evaluation.plots import plot_confusion_matrix, plot_pr_curve

import pandas as pd
import os
from collections import Counter

from src.utils.config import GCN_LR, GCN_EPOCHS


class EvolveGCN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.lin = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)

        return x


def train(model, data, mask, optimizer, weights):
    model.train()
    optimizer.zero_grad()

    out = model(data)

    loss = F.cross_entropy(
        out[mask],
        data.y[mask],
        weight=weights
    )

    loss.backward()
    optimizer.step()

    return loss.item()


def run_evolve_gcn(df_train, df_val, df_test, edges):

    print("\n===== EVOLVE GCN MODEL =====")

    model = None

    # ===== CLASS WEIGHTS =====
    class_counts = Counter(df_train["class"].values)
    total = sum(class_counts.values())

    weights = torch.tensor([
        total / class_counts[0],
        total / class_counts[1]
    ], dtype=torch.float)

    optimizer = None

    # ===== TREINO TEMPORAL =====
    for ts in sorted(df_train["time_step"].unique()):

        df_ts = df_train[df_train["time_step"] == ts]

        data = build_pyg_data(df_ts, edges)

        mask = torch.ones(data.num_nodes, dtype=torch.bool)

        if model is None:
            model = EvolveGCN(in_channels=data.num_features)
            optimizer = torch.optim.Adam(model.parameters(), lr=GCN_LR)

        loss = train(model, data, mask, optimizer, weights)

        if ts % 5 == 0:
            print(f"Time {ts} | Loss {loss:.4f}")

    # ===== VALIDAÇÃO =====
    df_val_all = df_val.copy()
    data_val = build_pyg_data(df_val_all, edges)

    val_mask = torch.ones(data_val.num_nodes, dtype=torch.bool)

    y_true_val, _, y_prob_val = evaluate(model, data_val, val_mask)
    val_pr_auc = average_precision_score(y_true_val, y_prob_val)

    print(f"\nVal PR-AUC: {val_pr_auc:.4f}")

    # ===== TESTE =====
    df_test_all = df_test.copy()
    data_test = build_pyg_data(df_test_all, edges)

    test_mask = torch.ones(data_test.num_nodes, dtype=torch.bool)

    y_true, y_pred, y_prob = evaluate(model, data_test, test_mask)

    results = {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "model": "EvolveGCN"
    }

    print("\n===== RESULTADOS EVOLVE GCN =====")
    print(results)

    os.makedirs("results/figures", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, "EvolveGCN")
    plot_pr_curve(y_true, y_prob, "EvolveGCN")

    os.makedirs("results/new_tables", exist_ok=True)
    pd.DataFrame([results]).to_csv("results/new_tables/evolve_gcn_results.csv", index=False)

    return results
