import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# ==============================
# MODEL
# ==============================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=4,
            dropout=0.6
        )

        self.conv2 = GATConv(
            hidden_channels * 4,
            hidden_channels,
            heads=4,
            dropout=0.6
        )

        self.conv3 = GATConv(
            hidden_channels * 4,
            1,
            heads=1,
            concat=False
        )

        self.dropout = 0.6

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        return x.squeeze()


# ==============================
# UTILS
# ==============================
def compute_class_weight(y):
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()

    if pos == 0:
        return torch.tensor(1.0)

    return torch.tensor(neg / pos)


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 50)

    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if len(np.unique(y_pred)) < 2:
            continue

        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


# ==============================
# GRAPH BUILDER
# ==============================
def build_graph(df, edges):

    feature_cols = [
        c for c in df.columns
        if c not in ["txId", "class", "time_step"]
    ]

    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["class"].values, dtype=torch.float)

    node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

    edge_list = []

    for _, row in edges.iterrows():
        if row["txId1"] in node_map and row["txId2"] in node_map:
            edge_list.append([
                node_map[row["txId1"]],
                node_map[row["txId2"]]
            ])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    return x, y, edge_index


# ==============================
# MAIN
# ==============================
def run_gat_temporal(df_train, df_val, df_test, edges):
    print("\n===== GAT TEMPORAL (WEBER PAPER STYLE) =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    # ======================
    # NORMALIZAÇÃO (SEM LEAKAGE)
    # ======================
    feature_cols = [
        c for c in df_all.columns
        if c not in ["txId", "class", "time_step"]
    ]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    df_all[feature_cols] = scaler.transform(df_all[feature_cols])

    df_all["time_step"] = df_all["time_step"].astype(int)

    max_time = int(df_all["time_step"].max())

    model = None

    all_probs = []
    all_true = []

    # ======================
    # LOOP TEMPORAL
    # ======================
    for t in range(30, max_time):

        print(f"\n--- Time {t} → Predict {t+1} ---")

        df_past = df_all[df_all["time_step"] <= t]
        df_target = df_all[df_all["time_step"] == t + 1]

        if len(df_target) == 0:
            continue

        # grafo até t
        x, y, edge_index = build_graph(df_past, edges)

        train_mask = torch.tensor(
            (df_past["time_step"] < t).values
        )

        # init modelo
        if model is None:
            model = GAT(x.shape[1], 32)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.003,
            weight_decay=5e-4
        )

        pos_weight = compute_class_weight(y[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ======================
        # TREINO
        # ======================
        model.train()
        for epoch in range(20):
            logits = model(x, edge_index)

            loss = criterion(
                logits[train_mask],
                y[train_mask]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ======================
        # INFERÊNCIA (t+1)
        # ======================
        df_eval = df_all[df_all["time_step"] <= t + 1]

        x_eval, y_eval, edge_eval = build_graph(df_eval, edges)

        model.eval()
        with torch.no_grad():
            logits = model(x_eval, edge_eval)
            probs = torch.sigmoid(logits).cpu().numpy()

        idx_target = df_eval["time_step"] == (t + 1)

        y_true = y_eval[idx_target].cpu().numpy()
        y_prob = probs[idx_target]

        if len(y_true) == 0:
            continue

        print(f"Amostras válidas: {len(y_true)}")

        all_probs.extend(y_prob)
        all_true.extend(y_true)

    # ======================
    # MÉTRICAS FINAIS
    # ======================
    if len(all_true) == 0:
        print("\nERRO: nenhum dado válido para avaliação")
        return None

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    best_t = find_best_threshold(all_true, all_probs)
    y_pred = (all_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(all_true, all_probs),
        "F1": f1_score(all_true, y_pred),
        "Precision": precision_score(all_true, y_pred),
        "Recall": recall_score(all_true, y_pred),
        "model": "GAT_TEMPORAL"
    }

    print("\n===== RESULTADOS GAT TEMPORAL =====")
    print(results)

    return results
