# =============== threshold dinâmico temporal ==========================
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import optuna

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops, degree

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve
)

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==============================
# TRAINING CONFIG
# ==============================
MAX_EPOCHS = 1000
PATIENCE = 20


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1, best_t = 0, 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t


# ==============================
# MODEL
# ==============================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, 1)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)

        x3 = self.conv3(x2, edge_index)

        return x3.squeeze()


# ==============================
# UTILS
# ==============================
def compute_class_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return torch.tensor(float(neg / pos), device=device) if pos > 0 else torch.tensor(1.0, device=device)


def build_graph(df, edges):
    feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

    x = torch.tensor(df[feature_cols].values, dtype=torch.float, device=device)
    y = torch.tensor(df["class"].values, dtype=torch.float, device=device)

    node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

    edge_list = [
        [node_map[row["txId1"]], node_map[row["txId2"]]]
        for _, row in edges.iterrows()
        if row["txId1"] in node_map and row["txId2"] in node_map
    ]

    if len(edge_list) == 0:
        return None, None, None, None

    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    deg = degree(edge_index[0], x.size(0)).view(-1, 1).to(device)
    x = torch.cat([x, deg], dim=1)

    return x, y, edge_index, node_map


# ==============================
# OPTUNA OBJECTIVE
# ==============================
def objective(trial, df_all, edges):

    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    scores = []

    for t in [32, 33, 34]:

        df_past = df_all[df_all["time_step"] <= t]

        x, y, edge_index, _ = build_graph(df_past, edges)
        if x is None:
            continue

        train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
        val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        model = GraphSAGE(x.shape[1], hidden_dim, dropout).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        pos_weight = compute_class_weight(y[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val = 0
        counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()

            logits = model(x, edge_index)
            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_val = model(x, edge_index)
                probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

            y_true = y[val_mask].cpu().numpy()

            if len(np.unique(y_true)) < 2:
                continue

            score = average_precision_score(y_true, probs)

            if score > best_val:
                best_val = score
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                break

        scores.append(best_val)

    return np.mean(scores) if len(scores) > 0 else 0


# ==============================
# TUNER
# ==============================
def tune_graphsage(df_train, df_val, edges):

    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, df_all, edges),
        n_trials=70
    )

    print("\n===== MELHORES PARÂMETROS GRAPHSAGE =====")
    print(study.best_params)

    return study.best_params


# ==============================
# MAIN
# ==============================
def run_graphsage_temporal(df_train, df_val, df_test, edges):

    print("\n===== GRAPHSAGE =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    feature_cols = [
        c for c in df_all.columns
        if c not in ["txId", "class", "time_step"]
    ]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    df_all[feature_cols] = scaler.transform(df_all[feature_cols])

    df_all["time_step"] = df_all["time_step"].astype(int)
    max_time = int(df_all["time_step"].max())

    best_params = tune_graphsage(df_train, df_val, edges)

    all_probs = []
    all_true = []
    all_preds = []

    for t in range(30, max_time):

        df_past = df_all[df_all["time_step"] <= t]

        x, y, edge_index, _ = build_graph(df_past, edges)
        if x is None:
            continue

        train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
        val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

        if train_mask.sum() == 0:
            continue

        model = GraphSAGE(
            x.shape[1],
            best_params["hidden_dim"],
            best_params["dropout"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"]
        )

        pos_weight = compute_class_weight(y[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val = 0
        counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()

            logits = model(x, edge_index)
            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_val = model(x, edge_index)
                probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

            y_true = y[val_mask].cpu().numpy()

            if len(np.unique(y_true)) < 2:
                continue

            score = average_precision_score(y_true, probs)

            if score > best_val:
                best_val = score
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                break

        model.eval()

        with torch.no_grad():
            logits_val = model(x, edge_index)
            val_probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

        val_true = y[val_mask].cpu().numpy()

        if len(np.unique(val_true)) >= 2:
            best_t = find_best_threshold(val_true, val_probs)
        else:
            best_t = 0.5

        df_eval = df_all[df_all["time_step"] <= t + 1]
        x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

        if x_eval is None:
            continue

        with torch.no_grad():
            logits_eval = model(x_eval, edge_eval)
            probs = torch.sigmoid(logits_eval).cpu().numpy()

        idx = (df_eval["time_step"] == (t + 1)).values

        if idx.sum() == 0:
            continue

        y_step_true = y_eval[idx].cpu().numpy()
        y_step_prob = probs[idx]
        y_step_pred = (y_step_prob >= best_t).astype(int)

        all_true.extend(y_step_true)
        all_probs.extend(y_step_prob)
        all_preds.extend(y_step_pred)

    y_true = np.array(all_true)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    results = {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "model": "GraphSAGE"
    }

    print("\n===== RESULTADOS GRAPHSAGE =====")
    print(results)

    plot_confusion_matrix(y_true, y_pred, "GraphSAGE")
    plot_pr_curve(y_true, y_prob, "GraphSAGE")

    return results


# ===============  threshold dinâmico e fixo  ==========================
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import to_undirected, add_self_loops, degree

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score
# )

# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# # ==============================
# # DEVICE
# # ==============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")

# # ==============================
# # TRAINING CONFIG
# # ==============================
# MAX_EPOCHS = 1000
# PATIENCE = 20


# def find_best_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.1, 0.9, 50)
#     best_f1, best_t = 0, 0.5

#     for t in thresholds:
#         preds = (y_prob >= t).astype(int)
#         f1 = f1_score(y_true, preds, zero_division=0)

#         if f1 > best_f1:
#             best_f1, best_t = f1, t

#     return best_t


# # ==============================
# # MODEL
# # ==============================
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout):
#         super().__init__()

#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv3 = SAGEConv(hidden_channels, 1)

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2 + x1)

#         x3 = self.conv3(x2, edge_index)

#         return x3.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return torch.tensor(float(neg / pos), device=device) if pos > 0 else torch.tensor(1.0, device=device)


# def build_graph(df, edges):
#     feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float, device=device)
#     y = torch.tensor(df["class"].values, dtype=torch.float, device=device)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = [
#         [node_map[row["txId1"]], node_map[row["txId2"]]]
#         for _, row in edges.iterrows()
#         if row["txId1"] in node_map and row["txId2"] in node_map
#     ]

#     if len(edge_list) == 0:
#         return None, None, None, None

#     edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     deg = degree(edge_index[0], x.size(0)).view(-1, 1).to(device)
#     x = torch.cat([x, deg], dim=1)

#     return x, y, edge_index, node_map


# # ==============================
# # OPTUNA OBJECTIVE
# # ==============================
# def objective(trial, df_all, edges):

#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
#     lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.5)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

#     scores = []

#     for t in [32, 33, 34]:

#         df_past = df_all[df_all["time_step"] <= t]

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
#         val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

#         if train_mask.sum() == 0 or val_mask.sum() == 0:
#             continue

#         model = GraphSAGE(x.shape[1], hidden_dim, dropout).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=lr,
#             weight_decay=weight_decay
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_val = 0
#         counter = 0

#         for epoch in range(MAX_EPOCHS):
#             model.train()

#             logits = model(x, edge_index)
#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 logits_val = model(x, edge_index)
#                 probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#             y_true = y[val_mask].cpu().numpy()

#             if len(np.unique(y_true)) < 2:
#                 continue

#             score = average_precision_score(y_true, probs)

#             if score > best_val:
#                 best_val = score
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= PATIENCE:
#                 break

#         scores.append(best_val)

#     return np.mean(scores) if len(scores) > 0 else 0


# # ==============================
# # TUNER
# # ==============================
# def tune_graphsage(df_train, df_val, edges):

#     df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges),
#         n_trials=70
#     )

#     print("\n===== MELHORES PARÂMETROS GRAPHSAGE =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN
# # ==============================
# def run_graphsage_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GRAPHSAGE =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])
#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     best_params = tune_graphsage(df_train, df_val, edges)

#     all_probs = []
#     all_true = []

#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
#         val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

#         if train_mask.sum() == 0:
#             continue

#         model = GraphSAGE(
#             x.shape[1],
#             best_params["hidden_dim"],
#             best_params["dropout"]
#         ).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=best_params["lr"],
#             weight_decay=best_params["weight_decay"]
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_val = 0
#         counter = 0

#         for epoch in range(MAX_EPOCHS):
#             model.train()

#             logits = model(x, edge_index)
#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 logits_val = model(x, edge_index)
#                 probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#             y_true = y[val_mask].cpu().numpy()

#             if len(np.unique(y_true)) < 2:
#                 continue

#             score = average_precision_score(y_true, probs)

#             if score > best_val:
#                 best_val = score
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= PATIENCE:
#                 break

#         model.eval()

#         # ===== THRESHOLD NA VALIDAÇÃO =====
#         with torch.no_grad():
#             logits_val = model(x, edge_index)
#             val_probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#         val_true = y[val_mask].cpu().numpy()

#         if len(np.unique(val_true)) >= 2:
#             best_t = find_best_threshold(val_true, val_probs)
#         else:
#             best_t = 0.5

#         # ===== APLICA NO PRÓXIMO TIME STEP =====
#         df_eval = df_all[df_all["time_step"] <= t + 1]
#         x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

#         if x_eval is None:
#             continue

#         with torch.no_grad():
#             logits_eval = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits_eval).cpu().numpy()

#         idx = (df_eval["time_step"] == (t + 1)).values

#         if idx.sum() == 0:
#             continue

#         all_true.extend(y_eval[idx].cpu().numpy())
#         all_probs.extend(probs[idx])

#     y_true = np.array(all_true)
#     y_prob = np.array(all_probs)

#     y_pred = (y_prob >= 0.5).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred, zero_division=0),
#         "Precision": precision_score(y_true, y_pred, zero_division=0),
#         "Recall": recall_score(y_true, y_pred, zero_division=0),
#         "model": "GraphSAGE"
#     }

#     print("\n===== RESULTADOS GRAPHSAGE =====")
#     print(results)

#     plot_confusion_matrix(y_true, y_pred, "GraphSAGE")
#     plot_pr_curve(y_true, y_prob, "GraphSAGE")

#     return results

# =============== threshold fixo ==========================
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import to_undirected, add_self_loops, degree

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score
# )

# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# # ==============================
# # DEVICE
# # ==============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")

# # ==============================
# # TRAINING CONFIG (ALTERADO)
# # ==============================
# MAX_EPOCHS = 1000
# PATIENCE = 20


# # ==============================
# # MODEL
# # ==============================
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout):
#         super().__init__()

#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv3 = SAGEConv(hidden_channels, 1)

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2 + x1)

#         x3 = self.conv3(x2, edge_index)

#         return x3.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return torch.tensor(float(neg / pos), device=device) if pos > 0 else torch.tensor(1.0, device=device)


# def build_graph(df, edges):
#     feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float, device=device)
#     y = torch.tensor(df["class"].values, dtype=torch.float, device=device)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = [
#         [node_map[row["txId1"]], node_map[row["txId2"]]]
#         for _, row in edges.iterrows()
#         if row["txId1"] in node_map and row["txId2"] in node_map
#     ]

#     if len(edge_list) == 0:
#         return None, None, None, None

#     edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     deg = degree(edge_index[0], x.size(0)).view(-1, 1).to(device)
#     x = torch.cat([x, deg], dim=1)

#     return x, y, edge_index, node_map


# # ==============================
# # OPTUNA OBJECTIVE
# # ==============================
# def objective(trial, df_all, edges):

#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
#     lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.5)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

#     scores = []

#     for t in [32, 33, 34]:

#         df_past = df_all[df_all["time_step"] <= t]

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
#         val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

#         if train_mask.sum() == 0 or val_mask.sum() == 0:
#             continue

#         model = GraphSAGE(x.shape[1], hidden_dim, dropout).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=lr,
#             weight_decay=weight_decay
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_val = 0
#         counter = 0

#         for epoch in range(MAX_EPOCHS):
#             model.train()

#             logits = model(x, edge_index)
#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 logits_val = model(x, edge_index)
#                 probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#             y_true = y[val_mask].cpu().numpy()

#             if len(np.unique(y_true)) < 2:
#                 continue

#             score = average_precision_score(y_true, probs)

#             if score > best_val:
#                 best_val = score
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= PATIENCE:
#                 break

#         scores.append(best_val)

#     return np.mean(scores) if len(scores) > 0 else 0


# # ==============================
# # TUNER
# # ==============================
# def tune_graphsage(df_train, df_val, edges):

#     df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges),
#         n_trials=70
#     )

#     print("\n===== MELHORES PARÂMETROS GRAPHSAGE =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN FINAL (ALTERADO EARLY STOP)
# # ==============================
# def run_graphsage_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GRAPHSAGE =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])
#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     best_params = tune_graphsage(df_train, df_val, edges)

#     all_probs = []
#     all_true = []

#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
#         val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

#         if train_mask.sum() == 0:
#             continue

#         model = GraphSAGE(
#             x.shape[1],
#             best_params["hidden_dim"],
#             best_params["dropout"]
#         ).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=best_params["lr"],
#             weight_decay=best_params["weight_decay"]
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_val = 0
#         counter = 0

#         for epoch in range(MAX_EPOCHS):
#             model.train()

#             logits = model(x, edge_index)
#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 logits_val = model(x, edge_index)
#                 probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#             y_true = y[val_mask].cpu().numpy()

#             if len(np.unique(y_true)) < 2:
#                 continue

#             score = average_precision_score(y_true, probs)

#             if score > best_val:
#                 best_val = score
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= PATIENCE:
#                 break

#         model.eval()

#         df_eval = df_all[df_all["time_step"] <= t + 1]
#         x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

#         if x_eval is None:
#             continue

#         with torch.no_grad():
#             logits_eval = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits_eval).cpu().numpy()

#         idx = (df_eval["time_step"] == (t + 1)).values

#         if idx.sum() == 0:
#             continue

#         all_true.extend(y_eval[idx].cpu().numpy())
#         all_probs.extend(probs[idx])

#     y_true = np.array(all_true)
#     y_prob = np.array(all_probs)

#     y_pred = (y_prob >= 0.5).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred, zero_division=0),
#         "Precision": precision_score(y_true, y_pred, zero_division=0),
#         "Recall": recall_score(y_true, y_pred, zero_division=0),
#         "model": "GraphSAGE"
#     }

#     print("\n===== RESULTADOS GRAPHSAGE =====")
#     print(results)

#     plot_confusion_matrix(y_true, y_pred, "GraphSAGE")
#     plot_pr_curve(y_true, y_prob, "GraphSAGE")

#     return results




















#===============================================================================================================

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import to_undirected, add_self_loops, degree

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score
# )

# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# # ==============================
# # DEVICE
# # ==============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")


# # ==============================
# # MODEL
# # ==============================
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout):
#         super().__init__()

#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv3 = SAGEConv(hidden_channels, 1)

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2 + x1)

#         x3 = self.conv3(x2, edge_index)

#         return x3.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return torch.tensor(float(neg / pos), device=device) if pos > 0 else torch.tensor(1.0, device=device)


# def build_graph(df, edges):
#     feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float, device=device)
#     y = torch.tensor(df["class"].values, dtype=torch.float, device=device)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = [
#         [node_map[row["txId1"]], node_map[row["txId2"]]]
#         for _, row in edges.iterrows()
#         if row["txId1"] in node_map and row["txId2"] in node_map
#     ]

#     if len(edge_list) == 0:
#         return None, None, None, None

#     edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     deg = degree(edge_index[0], x.size(0)).view(-1, 1)
#     deg = deg.to(device)

#     x = torch.cat([x, deg], dim=1)

#     return x, y, edge_index, node_map


# # ==============================
# # OPTUNA OBJECTIVE
# # ==============================
# def objective(trial, df_all, edges):

#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
#     lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.5)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

#     scores = []

#     for t in [32, 33, 34]:

#         df_past = df_all[df_all["time_step"] <= t]

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
#         val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

#         if train_mask.sum() == 0 or val_mask.sum() == 0:
#             continue

#         model = GraphSAGE(x.shape[1], hidden_dim, dropout).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=lr,
#             weight_decay=weight_decay
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_val = 0
#         patience = 5
#         counter = 0

#         for epoch in range(100):
#             model.train()

#             logits = model(x, edge_index)
#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 logits_val = model(x, edge_index)
#                 probs = torch.sigmoid(logits_val[val_mask]).detach().cpu().numpy()

#             y_true = y[val_mask].detach().cpu().numpy()
#             score = average_precision_score(y_true, probs)

#             if score > best_val:
#                 best_val = score
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= patience:
#                 break

#         scores.append(best_val)

#     return np.mean(scores) if len(scores) > 0 else 0


# # ==============================
# # TUNER
# # ==============================
# def tune_graphsage(df_train, df_val, edges):

#     df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges),
#         n_trials=50
#     )

#     print("\n===== MELHORES PARÂMETROS GRAPHSAGE =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN FINAL
# # ==============================
# def run_graphsage_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GRAPHSAGE  (GPU) =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])
#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     best_params = tune_graphsage(df_train, df_val, edges)

#     all_test_probs = []
#     all_test_true = []

#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)

#         if train_mask.sum() == 0:
#             continue

#         model = GraphSAGE(
#             x.shape[1],
#             best_params["hidden_dim"],
#             best_params["dropout"]
#         ).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=best_params["lr"],
#             weight_decay=best_params["weight_decay"]
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_loss = float("inf")
#         patience = 5
#         counter = 0

#         for epoch in range(100):
#             model.train()

#             logits = model(x, edge_index)
#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= patience:
#                 break

#         model.eval()

#         df_eval = df_all[df_all["time_step"] <= t + 1]
#         x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

#         if x_eval is None:
#             continue

#         with torch.no_grad():
#             logits_eval = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits_eval).detach().cpu().numpy()

#         idx = (df_eval["time_step"] == (t + 1)).values

#         if idx.sum() == 0:
#             continue

#         all_test_true.extend(y_eval[idx].detach().cpu().numpy())
#         all_test_probs.extend(probs[idx])

#     y_true = np.array(all_test_true)
#     y_prob = np.array(all_test_probs)

#     y_pred = (y_prob >= 0.5).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred, zero_division=0),
#         "Precision": precision_score(y_true, y_pred, zero_division=0),
#         "Recall": recall_score(y_true, y_pred, zero_division=0)
#     }

#     print("\n===== RESULTADOS GRAPHSAGE =====")
#     print(results)

#     plot_confusion_matrix(y_true, y_pred, "GraphSAGE_Turbo")
#     plot_pr_curve(y_true, y_prob, "GraphSAGE_Turbo")

#     return results






# ===============================================================================================================
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import to_undirected, add_self_loops

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score
# )

# # plots
# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )


# # ==============================
# # MODEL
# # ==============================
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout):
#         super().__init__()

#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv3 = SAGEConv(hidden_channels, 1)

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2 + x1)  # residual

#         x3 = self.conv3(x2, edge_index)

#         return x3.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return torch.tensor(float(neg / pos)) if pos > 0 else torch.tensor(1.0)


# def build_graph(df, edges):
#     feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float)
#     y = torch.tensor(df["class"].values, dtype=torch.float)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = [
#         [node_map[row["txId1"]], node_map[row["txId2"]]]
#         for _, row in edges.iterrows()
#         if row["txId1"] in node_map and row["txId2"] in node_map
#     ]

#     if len(edge_list) == 0:
#         return None, None, None, None

#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     return x, y, edge_index, node_map


# # ==============================
# # OPTUNA OBJECTIVE (SEM TESTE)
# # ==============================
# def objective(trial, df_all, edges, feature_cols):

#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
#     lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.5)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

#     scores = []

#     # múltiplos timesteps → mais robusto
#     for t in [32, 33, 34]:

#         df_past = df_all[df_all["time_step"] <= t]

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values)
#         val_mask = torch.tensor((df_past["time_step"] == t).values)

#         if train_mask.sum() == 0 or val_mask.sum() == 0:
#             continue

#         model = GraphSAGE(len(feature_cols), hidden_dim, dropout)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=lr,
#             weight_decay=weight_decay
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         # treino curto (Optuna)
#         for _ in range(30):
#             model.train()
#             logits = model(x, edge_index)

#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # validação
#         model.eval()
#         with torch.no_grad():
#             logits_val = model(x, edge_index)
#             probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#         y_true = y[val_mask].cpu().numpy()

#         score = average_precision_score(y_true, probs)
#         scores.append(score)

#     if len(scores) == 0:
#         return 0

#     return np.mean(scores)


# # ==============================
# # TUNER
# # ==============================
# def tune_graphsage(df_train, df_val, edges):

#     df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges, feature_cols),
#         n_trials=50
#     )

#     print("\n===== MELHORES PARÂMETROS GRAPHSAGE =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN FINAL
# # ==============================
# def run_graphsage_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GRAPHSAGE FINAL =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     # ==============================
#     # NORMALIZAÇÃO (SEM LEAKAGE)
#     # ==============================
#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])
#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     # ==============================
#     # OPTUNA (SEM TESTE)
#     # ==============================
#     best_params = tune_graphsage(df_train, df_val, edges)

#     all_test_probs = []
#     all_test_true = []

#     # ==============================
#     # LOOP TEMPORAL (CORRETO)
#     # ==============================
#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values)

#         if train_mask.sum() == 0:
#             continue

#         # REINICIALIZA MODELO A CADA T
#         model = GraphSAGE(
#             len(feature_cols),
#             best_params["hidden_dim"],
#             best_params["dropout"]
#         )

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=best_params["lr"],
#             weight_decay=best_params["weight_decay"]
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         # treino
#         for _ in range(50):
#             model.train()
#             logits = model(x, edge_index)

#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # ==========================
#         # AVALIAÇÃO EM t+1
#         # ==========================
#         model.eval()

#         df_eval = df_all[df_all["time_step"] <= t + 1]
#         x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

#         if x_eval is None:
#             continue

#         with torch.no_grad():
#             logits_eval = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits_eval).cpu().numpy()

#         idx = (df_eval["time_step"] == (t + 1)).values

#         if idx.sum() == 0:
#             continue

#         all_test_true.extend(y_eval[idx].cpu().numpy())
#         all_test_probs.extend(probs[idx])

#     # ==============================
#     # MÉTRICAS FINAIS
#     # ==============================
#     y_true = np.array(all_test_true)
#     y_prob = np.array(all_test_probs)

#     y_pred = (y_prob >= 0.5).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred, zero_division=0),
#         "Precision": precision_score(y_true, y_pred, zero_division=0),
#         "Recall": recall_score(y_true, y_pred, zero_division=0)
#     }

#     print("\n===== RESULTADOS GRAPHSAGE FINAL =====")
#     print(results)

#     # ==============================
#     # GRÁFICOS
#     # ==============================
#     plot_confusion_matrix(y_true, y_pred, "GraphSAGE")
#     plot_pr_curve(y_true, y_prob, "GraphSAGE")

#     return results


# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import to_undirected, add_self_loops

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# # plots
# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# # ==============================
# # MODEL
# # ==============================
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout):
#         super().__init__()

#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv3 = SAGEConv(hidden_channels, 1)

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2 + x1)  # residual

#         x3 = self.conv3(x2, edge_index)

#         return x3.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return torch.tensor(float(neg / pos)) if pos > 0 else torch.tensor(1.0)


# def find_best_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.1, 0.9, 50)
#     best_f1, best_t = 0, 0.5

#     for t in thresholds:
#         preds = (y_prob >= t).astype(int)
#         f1 = f1_score(y_true, preds, zero_division=0)

#         if f1 > best_f1:
#             best_f1, best_t = f1, t

#     return best_t


# # ==============================
# # GRAPH BUILDER
# # ==============================
# def build_graph(df, edges):
#     feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float)
#     y = torch.tensor(df["class"].values, dtype=torch.float)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = [
#         [node_map[row["txId1"]], node_map[row["txId2"]]]
#         for _, row in edges.iterrows()
#         if row["txId1"] in node_map and row["txId2"] in node_map
#     ]

#     if len(edge_list) == 0:
#         return None, None, None, None

#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     return x, y, edge_index, node_map


# # ==============================
# # OPTUNA OBJECTIVE
# # ==============================
# def objective(trial, df_all, edges, feature_cols):

#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
#     lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.6)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

#     t = 34  # timestep fixo

#     df_past = df_all[df_all["time_step"] <= t]

#     x, y, edge_index, _ = build_graph(df_past, edges)
#     if x is None:
#         return 0

#     train_mask = torch.tensor((df_past["time_step"] < t).values)
#     val_mask = torch.tensor((df_past["time_step"] == t).values)

#     model = GraphSAGE(len(feature_cols), hidden_dim, dropout)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#     pos_weight = compute_class_weight(y[train_mask])
#     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     best_val = 0

#     for epoch in range(50):
#         model.train()

#         logits = model(x, edge_index)
#         loss = criterion(logits[train_mask], y[train_mask])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # validação
#         model.eval()
#         with torch.no_grad():
#             logits_val = model(x, edge_index)
#             probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

#         y_true = y[val_mask].cpu().numpy()

#         score = average_precision_score(y_true, probs)

#         if score > best_val:
#             best_val = score

#     return best_val


# # ==============================
# # TUNER
# # ==============================
# def tune_graphsage(df_train, df_val, df_test, edges):

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges, feature_cols),
#         n_trials=30
#     )

#     print("\n===== MELHORES PARÂMETROS GRAPHSAGE =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN FINAL (COM OPTUNA)
# # ==============================
# def run_graphsage_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GRAPHSAGE + OPTUNA =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     # normalização
#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])
#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     # ==============================
#     # OPTUNA
#     # ==============================
#     best_params = tune_graphsage(df_train, df_val, df_test, edges)

#     model = GraphSAGE(
#         len(feature_cols),
#         best_params["hidden_dim"],
#         best_params["dropout"]
#     )

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=best_params["lr"],
#         weight_decay=best_params["weight_decay"]
#     )

#     all_test_probs = []
#     all_test_true = []

#     # ==============================
#     # LOOP TEMPORAL COMPLETO
#     # ==============================
#     for t in range(30, max_time + 1):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values)

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         # treino
#         for epoch in range(50):
#             model.train()
#             logits = model(x, edge_index)

#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # avaliação
#         model.eval()

#         df_eval = df_all[df_all["time_step"] <= t + 1]
#         x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

#         with torch.no_grad():
#             logits_eval = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits_eval).cpu().numpy()

#         idx = (df_eval["time_step"] == (t + 1)).values

#         all_test_true.extend(y_eval[idx].cpu().numpy())
#         all_test_probs.extend(probs[idx])

#     # ==============================
#     # MÉTRICAS
#     # ==============================
#     y_true = np.array(all_test_true)
#     y_prob = np.array(all_test_probs)

#     t = find_best_threshold(y_true, y_prob)
#     y_pred = (y_prob >= t).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred),
#         "Precision": precision_score(y_true, y_pred),
#         "Recall": recall_score(y_true, y_pred)
#     }

#     print("\n===== RESULTADOS GRAPHSAGE =====")
#     print(results)

#     # gráficos
#     plot_confusion_matrix(y_true, y_pred, "GraphSAGE_Optuna")
#     plot_pr_curve(y_true, y_prob, "GraphSAGE_Optuna")

#     return results




""""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# plots
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve
)

# ==============================
# MODEL
# ==============================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, 1)

        self.dropout = 0.5

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)  # residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)

        return x3.squeeze()


# ==============================
# UTILS
# ==============================
def compute_class_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()

    if pos == 0:
        return torch.tensor(1.0)

    return torch.tensor(float(neg / pos))


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
    feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

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
# MAIN PIPELINE
# ==============================
def run_graphsage_temporal(df_train, df_val, df_test, edges):
    print("\n===== GRAPHSAGE TEMPORAL (MELHORADO) =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    # ==============================
    # NORMALIZAÇÃO SEM LEAKAGE
    # ==============================
    feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    df_all[feature_cols] = scaler.transform(df_all[feature_cols])

    df_all["time_step"] = df_all["time_step"].astype(int)
    max_time = int(df_all["time_step"].max())

    # ==============================
    # MODEL + OPTIMIZER (FORA DO LOOP)
    # ==============================
    model = GraphSAGE(len(feature_cols), 128)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=5e-4
    )

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

        x, y, edge_index, _ = build_graph(df_past, edges)

        if x is None:
            continue

        train_mask = torch.tensor((df_past["time_step"] < t).values)
        val_mask = torch.tensor((df_past["time_step"] == t).values)

        if train_mask.sum() == 0:
            continue

        pos_weight = compute_class_weight(y[train_mask]).to(x.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ==========================
        # TREINAMENTO COM EARLY STOPPING
        # ==========================
        best_val_loss = float("inf")
        patience = 10
        counter = 0

        for epoch in range(80):
            model.train()

            logits = model(x, edge_index)
            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validação
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index)
                val_loss = criterion(val_logits[val_mask], y[val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

        # ==========================
        # AVALIAÇÃO EM t+1
        # ==========================
        model.eval()

        df_eval = df_all[df_all["time_step"] <= t + 1]
        x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

        if x_eval is None:
            continue

        with torch.no_grad():
            logits_eval = model(x_eval, edge_eval)
            probs = torch.sigmoid(logits_eval).cpu().numpy()

        idx_target = (df_eval["time_step"] == (t + 1)).values

        if idx_target.sum() == 0:
            continue

        y_true = y_eval[idx_target].cpu().numpy()
        y_prob = probs[idx_target]

        all_test_true.extend(y_true)
        all_test_probs.extend(y_prob)

    # ==============================
    # MÉTRICAS FINAIS
    # ==============================
    all_test_true = np.array(all_test_true)
    all_test_probs = np.array(all_test_probs)

    best_t = find_best_threshold(all_test_true, all_test_probs)
    y_pred = (all_test_probs >= best_t).astype(int)

    results = {
        "PR_AUC": average_precision_score(all_test_true, all_test_probs),
        "F1": f1_score(all_test_true, y_pred, zero_division=0),
        "Precision": precision_score(all_test_true, y_pred, zero_division=0),
        "Recall": recall_score(all_test_true, y_pred, zero_division=0),
        "model": "GraphSAGE_TEMPORAL"
    }

    print("\n===== RESULTADOS GRAPHSAGE TEMPORAL =====")
    print(results)

    # ==============================
    # GRÁFICOS
    # ==============================
    plot_confusion_matrix(all_test_true, y_pred, "GraphSAGE_Temporal")
    plot_pr_curve(all_test_true, all_test_probs, "GraphSAGE_Temporal")

    return results
"""



# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np

# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import to_undirected, add_self_loops

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# # ==============================
# # MODEL
# # ==============================
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()

#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

#         self.conv3 = SAGEConv(hidden_channels, 1)

#         self.dropout = 0.5

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)

#         return x.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()

#     if pos == 0:
#         return torch.tensor(1.0)

#     return torch.tensor(float(neg / pos))


# def find_best_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.1, 0.9, 50)

#     best_f1 = 0
#     best_t = 0.5

#     for t in thresholds:
#         y_pred = (y_prob >= t).astype(int)
#         f1 = f1_score(y_true, y_pred, zero_division=0)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_t = t

#     return best_t


# # ==============================
# # GRAPH BUILDER
# # ==============================
# def build_graph(df, edges):
#     feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float)
#     y = torch.tensor(df["class"].values, dtype=torch.float)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = []
#     for _, row in edges.iterrows():
#         if row["txId1"] in node_map and row["txId2"] in node_map:
#             edge_list.append([node_map[row["txId1"]], node_map[row["txId2"]]])

#     if len(edge_list) == 0:
#         return None, None, None, None

#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     return x, y, edge_index, node_map


# # ==============================
# # MAIN PIPELINE
# # ==============================
# def run_graphsage_temporal(df_train, df_val, df_test, edges):
#     print("\n===== GRAPHSAGE TEMPORAL (WEBER PAPER STYLE) =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     # ==============================
#     # NORMALIZAÇÃO SEM LEAKAGE
#     # ==============================
#     feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])
#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     model = None

#     all_test_probs = []
#     all_test_true = []

#     # ==============================
#     # LOOP TEMPORAL (CORE DO PAPER)
#     # ==============================
#     for t in range(30, max_time + 1):
#         print(f"\n--- Time {t} → Predict {t+1} ---")

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         x, y, edge_index, _ = build_graph(df_past, edges)

#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values)
#         val_mask = torch.tensor((df_past["time_step"] == t).values)

#         if train_mask.sum() == 0:
#             continue

#         if model is None:
#             model = GraphSAGE(x.shape[1], 64)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=0.005,
#             weight_decay=5e-4
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         # ==========================
#         # TREINAMENTO
#         # ==========================
#         model.train()
#         for epoch in range(20):
#             logits = model(x, edge_index)

#             loss = criterion(logits[train_mask], y[train_mask])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # ==========================
#         # AVALIAÇÃO EM t+1
#         # ==========================
#         model.eval()

#         df_eval = df_all[df_all["time_step"] <= t + 1]
#         x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

#         if x_eval is None:
#             continue

#         with torch.no_grad():
#             logits_eval = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits_eval).cpu().numpy()

#         idx_target = (df_eval["time_step"] == (t + 1)).values

#         if idx_target.sum() == 0:
#             continue

#         y_true = y_eval[idx_target].cpu().numpy()
#         y_prob = probs[idx_target]

#         if len(y_true) == 0:
#             continue

#         print(f"Amostras válidas: {len(y_true)}")

#         all_test_true.extend(y_true)
#         all_test_probs.extend(y_prob)

#     # ==============================
#     # MÉTRICAS FINAIS
#     # ==============================
#     if len(all_test_true) == 0:
#         print("\nERRO: nenhum dado válido para avaliação")
#         return None

#     all_test_true = np.array(all_test_true)
#     all_test_probs = np.array(all_test_probs)

#     best_t = find_best_threshold(all_test_true, all_test_probs)
#     y_pred = (all_test_probs >= best_t).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(all_test_true, all_test_probs),
#         "F1": f1_score(all_test_true, y_pred, zero_division=0),
#         "Precision": precision_score(all_test_true, y_pred, zero_division=0),
#         "Recall": recall_score(all_test_true, y_pred, zero_division=0),
#         "model": "GraphSAGE_TEMPORAL"
#     }

#     print("\n===== RESULTADOS GRAPHSAGE TEMPORAL =====")
#     print(results)

#     return results
