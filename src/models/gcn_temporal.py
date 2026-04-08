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
    y_valid = y[y != -1]

    pos = (y_valid == 1).sum()
    neg = (y_valid == 0).sum()

    if pos == 0:
        return torch.tensor(1.0)

    return torch.tensor(neg / pos, dtype=torch.float)


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 50)

    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


# ==============================
# GRAPH BUILDER
# ==============================
def build_graph(df, edges):

    feature_cols = [c for c in df.columns if c not in ["txId", "class"]]

    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["class"].values, dtype=torch.float)

    node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

    edge_list = []

    for _, row in edges.iterrows():
        if row["txId1"] in node_map and row["txId2"] in node_map:
            edge_list.append([node_map[row["txId1"]], node_map[row["txId2"]]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    return x, y, edge_index


# ==============================
# MAIN
# ==============================
def run_gcn_temporal(df_train, df_val, df_test, edges):
    print("\n===== GCN TEMPORAL (WEBER PAPER STYLE) =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    # garantir tipo
    df_all["time_step"] = df_all["time_step"].astype(int)

    # =========================
    # NORMALIZAÇÃO (SEM LEAKAGE)
    # =========================
    feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    df_all[feature_cols] = scaler.transform(df_all[feature_cols])

    # =========================
    # LOOP TEMPORAL
    # =========================
    model = None

    all_test_probs = []
    all_test_true = []

    max_time = int(df_all["time_step"].max())

    for t in range(30, max_time):
        print(f"\n--- Time {t} → Predict {t+1} ---")

        df_past = df_all[df_all["time_step"] <= t]
        df_target = df_all[df_all["time_step"] == (t + 1)]

        if len(df_target) == 0:
            print("Sem dados em t+1 → pulando")
            continue

        # =====================
        # BUILD GRAPH (até t)
        # =====================
        x, y, edge_index = build_graph(df_past, edges)

        train_mask = (df_past["time_step"].values < t)
        val_mask = (df_past["time_step"].values == t)

        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)

        # remover labels desconhecidos
        train_mask = train_mask & (y != -1)
        val_mask = val_mask & (y != -1)

        if train_mask.sum() == 0:
            print("Sem dados de treino válidos → pulando")
            continue

        # =====================
        # INIT MODEL
        # =====================
        if model is None:
            model = GCN(x.shape[1], 64)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        pos_weight = compute_class_weight(y[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # =====================
        # TRAIN
        # =====================
        model.train()

        for epoch in range(20):
            logits = model(x, edge_index)

            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # =====================
        # EVAL EM t+1
        # =====================
        df_eval = df_all[df_all["time_step"] <= (t + 1)]

        x_eval, y_eval, edge_eval = build_graph(df_eval, edges)

        model.eval()
        with torch.no_grad():
            logits_eval = model(x_eval, edge_eval)
            probs = torch.sigmoid(logits_eval).cpu().numpy()

        mask_target = (df_eval["time_step"] == (t + 1)) & (df_eval["class"] != -1)

        if mask_target.sum() == 0:
            print("Sem labels válidos em t+1 → pulando")
            continue

        y_true = y_eval[mask_target].cpu().numpy()
        y_prob = probs[mask_target]

        print(f"Amostras válidas: {len(y_true)}")

        all_test_probs.extend(y_prob)
        all_test_true.extend(y_true)

    # =========================
    # RESULTADOS
    # =========================
    if len(all_test_true) == 0:
        print("\nERRO: nenhum dado válido para avaliação")
        return None

    all_test_probs = np.array(all_test_probs)
    all_test_true = np.array(all_test_true)

    best_t = find_best_threshold(all_test_true, all_test_probs)
    y_pred = (all_test_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(all_test_true, all_test_probs),
        "F1": f1_score(all_test_true, y_pred, zero_division=0),
        "Precision": precision_score(all_test_true, y_pred, zero_division=0),
        "Recall": recall_score(all_test_true, y_pred, zero_division=0),
        "model": "GCN_TEMPORAL"
    }

    print("\n===== RESULTADOS GCN TEMPORAL =====")
    print(results)

    return results
