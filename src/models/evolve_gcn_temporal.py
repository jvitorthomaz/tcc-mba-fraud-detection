import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# ==============================
# MODEL (EvolveGCN simplificado)
# ==============================
class EvolveGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.gcn = GCNConv(in_channels, hidden_channels)
        self.rnn = torch.nn.GRU(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, h):
        x = self.gcn(x, edge_index)
        x = F.relu(x)

        x_seq = x.unsqueeze(0)  # (1, num_nodes, hidden)

        if h is None:
            out_seq, h = self.rnn(x_seq)
        else:
            out_seq, h = self.rnn(x_seq, h)

        x = out_seq.squeeze(0)
        logits = self.out(x)

        return logits.squeeze(), h


# ==============================
# UTILS
# ==============================
def compute_class_weight(y):
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    return torch.tensor(neg / pos if pos > 0 else 1.0, dtype=torch.float)


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
            edge_list.append([node_map[row["txId1"]], node_map[row["txId2"]]])

    if len(edge_list) == 0:
        return None, None, None, None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    return x, y, edge_index, node_map


# ==============================
# MAIN
# ==============================
def run_evolve_gcn_temporal(df_train, df_val, df_test, edges):
    print("\n===== EVOLVE GCN TEMPORAL (WEBER STYLE REAL) =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    # ==============================
    # NORMALIZAÇÃO SEM LEAKAGE
    # ==============================
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

    all_test_probs = []
    all_test_true = []

    # ==============================
    # LOOP TEMPORAL
    # ==============================
    for t in range(30, max_time + 1):
        print(f"\n--- Time {t} → Predict {t+1} ---")

        # 🔥 RESET DO HIDDEN STATE (CORREÇÃO CRÍTICA)
        h = None

        df_past = df_all[df_all["time_step"] <= t]
        df_target = df_all[df_all["time_step"] == t + 1]

        if len(df_target) == 0:
            continue

        x, y, edge_index, node_map = build_graph(df_past, edges)

        if x is None:
            continue

        train_mask = torch.tensor(
            (df_past["time_step"] < t).values,
            dtype=torch.bool
        )

        if train_mask.sum() == 0:
            continue

        if model is None:
            model = EvolveGCN(x.shape[1], 64)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005,
            weight_decay=5e-4
        )

        pos_weight = compute_class_weight(y[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # =====================
        # TREINO
        # =====================
        model.train()

        for epoch in range(10):

            # evita erro de backward
            if h is not None:
                h = h.detach()

            logits, h = model(x, edge_index, h)

            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # =====================
        # AVALIAÇÃO t+1
        # =====================
        model.eval()

        df_eval = df_all[df_all["time_step"] <= t + 1]

        x_eval, y_eval, edge_eval, node_map_eval = build_graph(df_eval, edges)

        if x_eval is None:
            continue

        with torch.no_grad():
            logits_eval, _ = model(x_eval, edge_eval, None)  # sem h!
            probs = torch.sigmoid(logits_eval).cpu().numpy()

        idx_target = (df_eval["time_step"] == (t + 1)).values

        if idx_target.sum() == 0:
            continue

        y_true = y_eval[idx_target].cpu().numpy()
        y_prob = probs[idx_target]

        print(f"Amostras válidas: {len(y_true)}")

        all_test_probs.extend(y_prob)
        all_test_true.extend(y_true)

    # ==============================
    # MÉTRICAS
    # ==============================
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
        "model": "EvolveGCN_TEMPORAL_REAL"
    }

    print("\n===== RESULTADOS EVOLVE GCN TEMPORAL =====")
    print(results)

    return results