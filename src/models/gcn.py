import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# ==============================
# MODEL
# ==============================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, 1)

        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x.squeeze()


# ==============================
# UTILS
# ==============================
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


# ==============================
# DATA PREPARATION (FULL GRAPH)
# ==============================
def prepare_full_graph(df_all, edges):

    feature_cols = [c for c in df_all.columns if c not in ["txId", "class"]]

    # NORMALIZAÇÃO (CRÍTICO)
    scaler = StandardScaler()
    x_np = scaler.fit_transform(df_all[feature_cols].values)

    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(df_all["class"].values, dtype=torch.float)

    # MAPA DE NÓS
    node_map = {tx: i for i, tx in enumerate(df_all["txId"].values)}

    edge_index = []
    for _, row in edges.iterrows():
        if row["txId1"] in node_map and row["txId2"] in node_map:
            edge_index.append([node_map[row["txId1"]], node_map[row["txId2"]]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # IMPORTANTE: grafo não direcionado + self loops
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    return x, y, edge_index


def create_masks(df_all):
    train_mask = (df_all["time_step"] <= 30).values
    val_mask = ((df_all["time_step"] > 30) & (df_all["time_step"] <= 34)).values
    test_mask = (df_all["time_step"] > 34).values

    return (
        torch.tensor(train_mask),
        torch.tensor(val_mask),
        torch.tensor(test_mask)
    )


# ==============================
# TRAINING
# ==============================
def run_gcn(df_train, df_val, df_test, edges):
    print("\n===== GCN (FULL GRAPH) =====")

    # junta tudo
    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    x, y, edge_index = prepare_full_graph(df_all, edges)

    train_mask, val_mask, test_mask = create_masks(df_all)

    model = GCN(in_channels=x.shape[1], hidden_channels=64)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=5e-4
    )

    pos_weight = compute_class_weight(y[train_mask])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1
    best_state = copy.deepcopy(model.state_dict())

    patience = 20
    counter = 0

    # ==============================
    # TRAIN LOOP
    # ==============================
    for epoch in range(1, 301):

        model.train()
        logits = model(x, edge_index)

        loss = criterion(logits[train_mask], y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validação
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()

            val_pr = average_precision_score(val_true, val_probs)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss {loss:.4f} | Val PR-AUC {val_pr:.4f}")

        # early stopping
        if val_pr > best_val:
            best_val = val_pr
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping!")
            break

    # ==============================
    # TEST
    # ==============================
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)

        val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
        test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()

    val_true = y[val_mask].cpu().numpy()
    test_true = y[test_mask].cpu().numpy()

    best_t = find_best_threshold(val_true, val_probs)
    y_pred = (test_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(test_true, test_probs),
        "F1": f1_score(test_true, y_pred),
        "Precision": precision_score(test_true, y_pred),
        "Recall": recall_score(test_true, y_pred),
        "model": "GCN"
    }

    print("\n===== RESULTADOS GCN =====")
    print(results)

    return results


# import copy

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
# import numpy as np


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, 1)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x.squeeze()


# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return torch.tensor(neg / pos, dtype=torch.float)


# def find_best_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.1, 0.9, 50)
#     best_f1 = 0
#     best_t = 0.5

#     for t in thresholds:
#         y_pred = (y_prob >= t).astype(int)
#         f1 = f1_score(y_true, y_pred)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_t = t

#     return best_t


# def prepare_data(df, edges):
#     feature_cols = [col for col in df.columns if col not in ["txId", "class"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float)
#     y = torch.tensor(df["class"].values, dtype=torch.float)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_index = []
#     for _, row in edges.iterrows():
#         if row["txId1"] in node_map and row["txId2"] in node_map:
#             edge_index.append([node_map[row["txId1"]], node_map[row["txId2"]]])

#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

#     return x, y, edge_index


# def run_gcn(df_train, df_val, df_test, edges):
#     print("\n===== GCN MODEL =====")

#     x_train, y_train, edge_train = prepare_data(df_train, edges)
#     x_val, y_val, edge_val = prepare_data(df_val, edges)
#     x_test, y_test, edge_test = prepare_data(df_test, edges)

#     model = GCN(x_train.shape[1], 64)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     pos_weight = compute_class_weight(y_train)
#     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     best_val = -1
#     best_state = copy.deepcopy(model.state_dict())
#     patience = 10
#     counter = 0

#     for epoch in range(1, 201):
#         model.train()
#         logits = model(x_train, edge_train)
#         loss = criterion(logits, y_train)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_probs = torch.sigmoid(model(x_val, edge_val)).numpy()
#             val_pr = average_precision_score(y_val.numpy(), val_probs)

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch} | Loss {loss:.4f} | Val PR-AUC {val_pr:.4f}")

#         if val_pr > best_val:
#             best_val = val_pr
#             best_state = copy.deepcopy(model.state_dict())
#             counter = 0
#         else:
#             counter += 1

#         if counter >= patience:
#             print("Early stopping!")
#             break

#     model.load_state_dict(best_state)

#     with torch.no_grad():
#         test_probs = torch.sigmoid(model(x_test, edge_test)).numpy()

#     best_t = find_best_threshold(y_val.numpy(), val_probs)
#     y_pred = (test_probs >= best_t).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(y_test.numpy(), test_probs),
#         "F1": f1_score(y_test.numpy(), y_pred),
#         "Precision": precision_score(y_test.numpy(), y_pred),
#         "Recall": recall_score(y_test.numpy(), y_pred),
#         "model": "GCN"
#     }

#     print("\n===== RESULTADOS GCN =====")
#     print(results)

#     return results
