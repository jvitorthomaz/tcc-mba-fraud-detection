''''

python -m experiments.run_graph_transformer

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

from src.models.preliminary.gcn import build_pyg_data
from src.evaluation.plots import plot_confusion_matrix, plot_pr_curve

import pandas as pd
import os
from collections import Counter

from src.utils.config import GCN_LR, GCN_EPOCHS


# ==============================
# 1. MODELO TRANSFORMER
# ==============================

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        # x: [N, F]

        h = self.input_proj(x)        # [N, hidden]
        h = h.unsqueeze(0)           # [1, N, hidden]

        attn_out, _ = self.attention(h, h, h)

        out = self.mlp(attn_out.squeeze(0))  # [N, 2]

        return out


# ==============================
# 2. EVALUATION
# ==============================

def evaluate(model, data, df_all, df_split):

    model.eval()

    ts_values = df_split["time_step"].unique()
    mask = df_all["time_step"].isin(ts_values).values

    with torch.no_grad():
        logits = model(data.x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    y_true = data.y.cpu().numpy()

    return y_true[mask], preds[mask], probs[mask]


# ==============================
# 3. PIPELINE
# ==============================

def run_graph_transformer(df_train, df_val, df_test, edges):

    print("\n===== GRAPH TRANSFORMER MODEL =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    data = build_pyg_data(df_all, edges)

    model = GraphTransformer(in_channels=data.num_features)

    optimizer = torch.optim.Adam(model.parameters(), lr=GCN_LR)

    # ===== CLASS WEIGHTS =====
    class_counts = Counter(df_train["class"].values)
    total = sum(class_counts.values())

    weights = torch.tensor([
        total / class_counts[0],
        total / class_counts[1]
    ], dtype=torch.float)

    # ==============================
    # TREINO
    # ==============================
    for epoch in range(1, GCN_EPOCHS + 1):

        model.train()
        optimizer.zero_grad()

        logits = model(data.x)

        train_mask = df_all["time_step"] <= df_train["time_step"].max()

        loss = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
            weight=weights
        )

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            y_true_val, _, y_prob_val = evaluate(model, data, df_all, df_val)
            val_pr_auc = average_precision_score(y_true_val, y_prob_val)

            print(f"Epoch {epoch} | Loss {loss.item():.4f} | Val PR-AUC {val_pr_auc:.4f}")

    # ==============================
    # TESTE
    # ==============================
    y_true, y_pred, y_prob = evaluate(model, data, df_all, df_test)

    results = {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "model": "GraphTransformer"
    }

    print("\n===== RESULTADOS GRAPH TRANSFORMER =====")
    print(results)

    # ==============================
    # PLOTS
    # ==============================
    os.makedirs("results/figures", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, "GraphTransformer")
    plot_pr_curve(y_true, y_prob, "GraphTransformer")

    # ==============================
    # SALVAR CSV
    # ==============================
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame([results]).to_csv(
        "results/tables/graph_transformer_results.csv",
        index=False
    )

    print("\nResultados salvos em: results/tables/graph_transformer_results.csv")

    return results
