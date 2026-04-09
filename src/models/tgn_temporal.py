import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# ==============================
# MODEL
# ==============================
class TGN(nn.Module):
    def __init__(self, in_channels, memory_dim=64):
        super().__init__()

        self.memory_dim = memory_dim

        # projeta features → memória inicial
        self.input_proj = nn.Linear(in_channels, memory_dim)

        # message function
        self.message_mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + 1, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )

        # update memory
        self.gru = nn.GRUCell(memory_dim, memory_dim)

        # prediction
        self.classifier = nn.Linear(memory_dim, 1)

    def forward(self, memory):
        return self.classifier(memory).squeeze()


# ==============================
# UTILS
# ==============================
def compute_class_weight(y):
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
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
# BUILD GRAPH
# ==============================
def build_node_map(df):
    return {tx: i for i, tx in enumerate(df["txId"].values)}


def build_edge_index(edges, node_map):
    edge_list = []

    for _, row in edges.iterrows():
        if row["txId1"] in node_map and row["txId2"] in node_map:
            edge_list.append([node_map[row["txId1"]], node_map[row["txId2"]]])

    return edge_list


# ==============================
# MAIN TRAINING
# ==============================
def run_tgn_temporal(df_train, df_val, df_test, edges):

    print("\n===== TGN TEMPORAL (WEBER STYLE REAL) =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    # ==============================
    # NORMALIZAÇÃO (SEM LEAKAGE)
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

    # ==============================
    # INIT MODEL
    # ==============================
    model = TGN(in_channels=len(feature_cols), memory_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # ==============================
    # MEMORY GLOBAL (cresce dinamicamente)
    # ==============================
    global_node_map = {}
    memory = None
    last_update = None

    all_test_probs = []
    all_test_true = []

    # ==============================
    # LOOP TEMPORAL
    # ==============================
    for t in range(30, max_time + 1):

        print(f"\n--- Time {t} → Predict {t+1} ---")

        df_past = df_all[df_all["time_step"] <= t]
        df_target = df_all[df_all["time_step"] == t + 1]

        if len(df_target) == 0:
            continue

        # ==============================
        # BUILD NODE MAP GLOBAL
        # ==============================
        new_nodes = set(df_past["txId"].values).union(
            set(df_target["txId"].values)
        ) - set(global_node_map.keys())

        for tx in new_nodes:
            global_node_map[tx] = len(global_node_map)

        num_nodes = len(global_node_map)

        # expand memory
        if memory is None:
            memory = torch.zeros(num_nodes, model.memory_dim)
            last_update = torch.zeros(num_nodes)
        else:
            if num_nodes > memory.shape[0]:
                diff = num_nodes - memory.shape[0]
                memory = torch.cat([memory, torch.zeros(diff, model.memory_dim)], dim=0)
                last_update = torch.cat([last_update, torch.zeros(diff)])
        
        for _, row in df_target.iterrows():
          idx = global_node_map[row["txId"]]

          if torch.all(memory[idx] == 0):
              feat = torch.tensor(row[feature_cols].values, dtype=torch.float)
              memory[idx] = model.input_proj(feat)

        # ==============================
        # INITIALIZE NEW NODES
        # ==============================
        for _, row in df_past.iterrows():
            idx = global_node_map[row["txId"]]
            if torch.all(memory[idx] == 0):
                feat = torch.tensor(row[feature_cols].values, dtype=torch.float)
                memory[idx] = model.input_proj(feat)

        # ==============================
        # PROCESS EDGES (MESSAGE PASSING)
        # ==============================
        edge_list = build_edge_index(edges, global_node_map)

        new_memory = memory.clone()

        for src, dst in edge_list:

            delta_t = (t - last_update[src]).unsqueeze(0)

            msg_input = torch.cat([
                memory[src],
                memory[dst],
                delta_t
            ])

            msg = model.message_mlp(msg_input)

            new_memory[src] = model.gru(msg, memory[src])
            new_memory[dst] = model.gru(msg, memory[dst])

            last_update[src] = t
            last_update[dst] = t

        memory = new_memory.detach()

        # ==============================
        # TREINO (somente passado)
        # ==============================
        train_mask = df_past["time_step"] < t

        if train_mask.sum() == 0:
            continue

        train_nodes = [
            global_node_map[tx]
            for tx in df_past[train_mask]["txId"].values
        ]

        y_train = torch.tensor(
            df_past[train_mask]["class"].values,
            dtype=torch.float
        )

        logits = model(memory[train_nodes])

        pos_weight = compute_class_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ==============================
        # PREDIÇÃO t+1
        # ==============================
        target_nodes = [
            global_node_map[tx]
            for tx in df_target["txId"].values
        ]

        with torch.no_grad():
            logits_test = model(memory[target_nodes])
            probs = torch.sigmoid(logits_test).cpu().numpy()

        y_true = df_target["class"].values

        print(f"Amostras válidas: {len(y_true)}")

        all_test_probs.extend(probs)
        all_test_true.extend(y_true)

    # ==============================
    # MÉTRICAS
    # ==============================
    all_test_probs = np.array(all_test_probs)
    all_test_true = np.array(all_test_true)

    if len(all_test_true) == 0:
        print("\nERRO: nenhum dado válido para avaliação")
        return

    best_t = find_best_threshold(all_test_true, all_test_probs)
    y_pred = (all_test_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(all_test_true, all_test_probs),
        "F1": f1_score(all_test_true, y_pred),
        "Precision": precision_score(all_test_true, y_pred),
        "Recall": recall_score(all_test_true, y_pred),
        "model": "TGN_TEMPORAL_REAL"
    }

    print("\n===== RESULTADOS TGN TEMPORAL =====")
    print(results)

    return results
