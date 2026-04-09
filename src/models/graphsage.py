import copy

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import numpy as np


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()


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


def prepare_data(df, edges):
    feature_cols = [col for col in df.columns if col not in ["txId", "class"]]

    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["class"].values, dtype=torch.float)

    node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

    edge_index = []
    for _, row in edges.iterrows():
        if row["txId1"] in node_map and row["txId2"] in node_map:
            edge_index.append([node_map[row["txId1"]], node_map[row["txId2"]]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, y, edge_index


def run_graphsage(df_train, df_val, df_test, edges):
    print("\n===== GRAPHSAGE MODEL =====")

    x_train, y_train, edge_train = prepare_data(df_train, edges)
    x_val, y_val, edge_val = prepare_data(df_val, edges)
    x_test, y_test, edge_test = prepare_data(df_test, edges)

    model = GraphSAGE(x_train.shape[1], 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pos_weight = compute_class_weight(y_train)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1
    best_state = copy.deepcopy(model.state_dict())
    patience = 10
    counter = 0

    for epoch in range(1, 201):
        model.train()

        logits = model(x_train, edge_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val, edge_val)).numpy()
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
        test_probs = torch.sigmoid(model(x_test, edge_test)).numpy()

    best_t = find_best_threshold(y_val.numpy(), val_probs)
    y_pred = (test_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(y_test.numpy(), test_probs),
        "F1": f1_score(y_test.numpy(), y_pred),
        "Precision": precision_score(y_test.numpy(), y_pred),
        "Recall": recall_score(y_test.numpy(), y_pred),
        "model": "GraphSAGE"
    }

    print("\n===== RESULTADOS GRAPHSAGE =====")
    print(results)

    return results


# import torch
# import torch.nn.functional as F

# from torch_geometric.nn import SAGEConv
# from torch_geometric.data import Data

# from sklearn.metrics import (
#     f1_score,
#     precision_score,
#     recall_score,
#     average_precision_score
# )

# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# import pandas as pd
# import numpy as np
# import os

# from src.utils.config import GCN_LR, GCN_EPOCHS


# # ==============================
# # 1. CONSTRUIR DATASET
# # ==============================

# def build_pyg_data(df, edges):
#     df = df.reset_index(drop=True)

#     id_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     X = df.drop(columns=["txId", "class"]).values
#     X = torch.tensor(X, dtype=torch.float)

#     y = torch.tensor(df["class"].values, dtype=torch.long)

#     edge_index = edges.copy()
#     edge_index = edge_index[
#         edge_index["txId1"].isin(id_map) & edge_index["txId2"].isin(id_map)
#     ]

#     edge_index = edge_index.replace({"txId1": id_map, "txId2": id_map})
#     edge_index = torch.tensor(edge_index.values.T, dtype=torch.long)

#     return Data(x=X, edge_index=edge_index, y=y)


# # ==============================
# # 2. MODELO GRAPHSAGE
# # ==============================

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, 64)
#         self.conv2 = SAGEConv(64, 32)
#         self.lin = torch.nn.Linear(32, 2)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         x = self.lin(x)

#         return x


# # ==============================
# # 3. TREINAMENTO
# # ==============================

# def train(model, data, train_mask, optimizer):
#     model.train()
#     optimizer.zero_grad()

#     out = model(data)

#     loss = F.cross_entropy(out[train_mask], data.y[train_mask])
#     loss.backward()
#     optimizer.step()

#     return loss.item()


# # ==============================
# # 4. AVALIAÇÃO
# # ==============================

# def evaluate(model, data, mask):
#     model.eval()

#     with torch.no_grad():
#         logits = model(data)

#         probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
#         preds = logits.argmax(dim=1).cpu().numpy()
#         y_true = data.y.cpu().numpy()

#     mask = mask.cpu().numpy()

#     y_true = y_true[mask]
#     preds = preds[mask]
#     probs = probs[mask]

#     return y_true, preds, probs


# # ==============================
# # 5. PIPELINE GRAPHSAGE
# # ==============================

# def run_graphsage(df_train, df_val, df_test, edges):

#     print("\n===== GRAPHSAGE MODEL =====")

#     df_all = pd.concat([df_train, df_val, df_test])

#     data = build_pyg_data(df_all, edges)

#     n_train = len(df_train)
#     n_val = len(df_val)

#     train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

#     train_mask[:n_train] = True
#     val_mask[n_train:n_train+n_val] = True
#     test_mask[n_train+n_val:] = True

#     model = GraphSAGE(in_channels=data.num_features)
#     optimizer = torch.optim.Adam(model.parameters(), lr=GCN_LR)

#     # ==============================
#     # TREINO
#     # ==============================

#     for epoch in range(1, GCN_EPOCHS + 1):
#         loss = train(model, data, train_mask, optimizer)

#         if epoch % 10 == 0:
#             y_true_val, y_pred_val, y_prob_val = evaluate(model, data, val_mask)

#             val_pr_auc = average_precision_score(y_true_val, y_prob_val)

#             print(f"Epoch {epoch} | Loss {loss:.4f} | Val PR-AUC {val_pr_auc:.4f}")

#     # ==============================
#     # TESTE FINAL
#     # ==============================

#     y_true, y_pred, y_prob = evaluate(model, data, test_mask)

#     test_metrics = {
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred, zero_division=0),
#         "Precision": precision_score(y_true, y_pred, zero_division=0),
#         "Recall": recall_score(y_true, y_pred),
#         "model": "GraphSAGE"
#     }

#     print("\n===== RESULTADOS GRAPHSAGE =====")
#     print(test_metrics)

#     # ==============================
#     # PLOTS
#     # ==============================

#     os.makedirs("results/figures", exist_ok=True)

#     plot_confusion_matrix(y_true, y_pred, "GraphSAGE")
#     plot_pr_curve(y_true, y_prob, "GraphSAGE")

#     # ==============================
#     # SALVAR CSV
#     # ==============================

#     os.makedirs("results/tables", exist_ok=True)

#     results_df = pd.DataFrame([test_metrics])
#     results_df.to_csv("results/tables/graphsage_results.csv", index=False)

#     print("\nResultados salvos em: results/tables/graphsage_results.csv")

#     return results_df
