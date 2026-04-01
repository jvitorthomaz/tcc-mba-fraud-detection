import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

from src.models.gcn import build_pyg_data
from src.evaluation.plots import plot_confusion_matrix, plot_pr_curve

import pandas as pd
import os
from collections import Counter

from src.utils.config import GCN_LR, GCN_EPOCHS


# ==============================
# 1. MODELO TGN CORRIGIDO
# ==============================

class TGN(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()

        self.memory_dim = hidden_dim

        # memória por nó
        self.memory = None

        # atualiza memória
        self.gru = nn.GRUCell(in_channels, hidden_dim)

        # agora usa features + memória
        self.lin = nn.Linear(in_channels + hidden_dim, 2)

    def init_memory(self, num_nodes):
        self.memory = torch.zeros((num_nodes, self.memory_dim))

    def forward(self, x):
        # concatena informação atual + histórica
        h = torch.cat([x, self.memory], dim=1)
        out = self.lin(h)
        return out

    def update_memory(self, x, node_indices):
        updated_memory = self.gru(
            x,
            self.memory[node_indices]
        )

        # MUITO IMPORTANTE
        self.memory[node_indices] = updated_memory.detach()


# ==============================
# 2. EVALUATION
# ==============================

def evaluate_tgn(model, data, df_all, df_split):

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

def run_tgn(df_train, df_val, df_test, edges):

    print("\n===== TGN MODEL =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    data = build_pyg_data(df_all, edges)

    model = TGN(in_channels=data.num_features)
    model.init_memory(data.num_nodes)

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

        total_loss = 0

        for ts in sorted(df_train["time_step"].unique()):

            idx = df_all[df_all["time_step"] == ts].index.values

            logits = model(data.x)

            loss = F.cross_entropy(
                logits[idx],
                data.y[idx],
                weight=weights
            )

            total_loss += loss

            # atualizar memória
            x = data.x[idx]
            model.update_memory(x, idx)

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            y_true_val, _, y_prob_val = evaluate_tgn(model, data, df_all, df_val)
            val_pr_auc = average_precision_score(y_true_val, y_prob_val)

            print(f"Epoch {epoch} | Loss {total_loss.item():.4f} | Val PR-AUC {val_pr_auc:.4f}")

    # ==============================
    # TESTE
    # ==============================
    y_true, y_pred, y_prob = evaluate_tgn(model, data, df_all, df_test)

    results = {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "model": "TGN"
    }

    print("\n===== RESULTADOS TGN =====")
    print(results)

    # ==============================
    # PLOTS
    # ==============================
    os.makedirs("results/figures", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, "TGN")
    plot_pr_curve(y_true, y_prob, "TGN")

    # ==============================
    # SALVAR CSV
    # ==============================
    os.makedirs("results/new_tables", exist_ok=True)
    pd.DataFrame([results]).to_csv("results/new_tables/tgn_results.csv", index=False)

    print("\nResultados salvos em: results/new_tables/tgn_results.csv")

    return results
