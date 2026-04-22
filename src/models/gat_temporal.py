# =============== threshold dinâmico temporal ==========================
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import optuna

from torch_geometric.nn import GATConv
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
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, heads):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)

        self.conv3 = GATConv(
            hidden_channels * heads,
            1,
            heads=1,
            concat=False
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2 + x1)

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

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    heads = trial.suggest_categorical("heads", [2, 4])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

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

        model = GAT(x.shape[1], hidden_dim, dropout, heads).to(device)

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
                probs = torch.sigmoid(logits_val[val_mask]).detach().cpu().numpy()

            y_true = y[val_mask].detach().cpu().numpy()
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
def tune_gat(df_train, df_val, edges):

    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, df_all, edges),
        n_trials=50
    )

    print("\n===== MELHORES PARÂMETROS GAT =====")
    print(study.best_params)

    return study.best_params


# ==============================
# MAIN FINAL
# ==============================
def run_gat_temporal(df_train, df_val, df_test, edges):

    print("\n===== GAT  (GPU) =====")

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

    best_params = tune_gat(df_train, df_val, edges)

    all_test_probs = []
    all_test_true = []
    all_test_preds = []

    for t in range(30, max_time):

        df_past = df_all[df_all["time_step"] <= t]

        x, y, edge_index, _ = build_graph(df_past, edges)
        if x is None:
            continue

        train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
        val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

        if train_mask.sum() == 0:
            continue

        model = GAT(
            x.shape[1],
            best_params["hidden_dim"],
            best_params["dropout"],
            best_params["heads"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"]
        )

        pos_weight = compute_class_weight(y[train_mask])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()

            logits = model(x, edge_index)
            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                break

        model.eval()

        # ===== THRESHOLD NA VALIDAÇÃO =====
        with torch.no_grad():
            logits_val = model(x, edge_index)
            val_probs = torch.sigmoid(logits_val[val_mask]).cpu().numpy()

        val_true = y[val_mask].cpu().numpy()

        if len(np.unique(val_true)) >= 2:
            best_t = find_best_threshold(val_true, val_probs)
        else:
            best_t = 0.5

        # ===== APLICA NO PRÓXIMO TIME STEP =====
        df_eval = df_all[df_all["time_step"] <= t + 1]
        x_eval, y_eval, edge_eval, _ = build_graph(df_eval, edges)

        if x_eval is None:
            continue

        with torch.no_grad():
            logits_eval = model(x_eval, edge_eval)
            probs = torch.sigmoid(logits_eval).detach().cpu().numpy()

        idx = (df_eval["time_step"] == (t + 1)).values

        if idx.sum() == 0:
            continue

        y_step_true = y_eval[idx].detach().cpu().numpy()
        y_step_prob = probs[idx]
        y_step_pred = (y_step_prob >= best_t).astype(int)

        all_test_true.extend(y_step_true)
        all_test_probs.extend(y_step_prob)
        all_test_preds.extend(y_step_pred)

    y_true = np.array(all_test_true)
    y_prob = np.array(all_test_probs)
    y_pred = np.array(all_test_preds)

    results = {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0)
    }

    print("\n===== RESULTADOS GAT =====")
    print(results)

    plot_confusion_matrix(y_true, y_pred, "GAT")
    plot_pr_curve(y_true, y_prob, "GAT")

    return results

# ===============  threshold dinâmico e fixo  ==========================
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import GATConv
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
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout, heads):
#         super().__init__()

#         self.conv1 = GATConv(
#             in_channels,
#             hidden_channels,
#             heads=heads,
#             dropout=dropout
#         )

#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)

#         self.conv2 = GATConv(
#             hidden_channels * heads,
#             hidden_channels,
#             heads=heads,
#             dropout=dropout
#         )

#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)

#         self.conv3 = GATConv(
#             hidden_channels * heads,
#             1,
#             heads=1,
#             concat=False
#         )

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.elu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.elu(x2 + x1)

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

#     hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
#     heads = trial.suggest_categorical("heads", [2, 4])
#     lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.6)
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

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

#         model = GAT(x.shape[1], hidden_dim, dropout, heads).to(device)

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
#                 probs = torch.sigmoid(logits_val[val_mask]).detach().cpu().numpy()

#             y_true = y[val_mask].detach().cpu().numpy()
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
# def tune_gat(df_train, df_val, edges):

#     df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges),
#         n_trials=50
#     )

#     print("\n===== MELHORES PARÂMETROS GAT =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN FINAL
# # ==============================
# def run_gat_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GAT  (GPU) =====")

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

#     best_params = tune_gat(df_train, df_val, edges)

#     all_test_probs = []
#     all_test_true = []

#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]

#         x, y, edge_index, _ = build_graph(df_past, edges)
#         if x is None:
#             continue

#         train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
#         val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

#         if train_mask.sum() == 0:
#             continue

#         model = GAT(
#             x.shape[1],
#             best_params["hidden_dim"],
#             best_params["dropout"],
#             best_params["heads"]
#         ).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=best_params["lr"],
#             weight_decay=best_params["weight_decay"]
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_loss = float("inf")
#         counter = 0

#         for epoch in range(MAX_EPOCHS):
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

#     print("\n===== RESULTADOS GAT =====")
#     print(results)

#     plot_confusion_matrix(y_true, y_pred, "GAT")
#     plot_pr_curve(y_true, y_prob, "GAT")

#     return results


# =============== threshold fixo ==========================
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import optuna

# from torch_geometric.nn import GATConv
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
# # TRAINING CONFIG (AJUSTADO)
# # ==============================
# MAX_EPOCHS = 1000
# PATIENCE = 20


# # ==============================
# # MODEL
# # ==============================
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout, heads):
#         super().__init__()

#         self.conv1 = GATConv(
#             in_channels,
#             hidden_channels,
#             heads=heads,
#             dropout=dropout
#         )

#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)

#         self.conv2 = GATConv(
#             hidden_channels * heads,
#             hidden_channels,
#             heads=heads,
#             dropout=dropout
#         )

#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)

#         self.conv3 = GATConv(
#             hidden_channels * heads,
#             1,
#             heads=1,
#             concat=False
#         )

#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x1 = self.conv1(x, edge_index)
#         x1 = self.bn1(x1)
#         x1 = F.elu(x1)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         x2 = self.conv2(x1, edge_index)
#         x2 = self.bn2(x2)
#         x2 = F.elu(x2 + x1)

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

#     hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
#     heads = trial.suggest_categorical("heads", [2, 4])
#     lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     dropout = trial.suggest_float("dropout", 0.2, 0.6)
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

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

#         model = GAT(x.shape[1], hidden_dim, dropout, heads).to(device)

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
#                 probs = torch.sigmoid(logits_val[val_mask]).detach().cpu().numpy()

#             y_true = y[val_mask].detach().cpu().numpy()
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
# def tune_gat(df_train, df_val, edges):

#     df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

#     study = optuna.create_study(direction="maximize")

#     study.optimize(
#         lambda trial: objective(trial, df_all, edges),
#         n_trials=50
#     )

#     print("\n===== MELHORES PARÂMETROS GAT =====")
#     print(study.best_params)

#     return study.best_params


# # ==============================
# # MAIN FINAL
# # ==============================
# def run_gat_temporal(df_train, df_val, df_test, edges):

#     print("\n===== GAT  (GPU) =====")

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

#     best_params = tune_gat(df_train, df_val, edges)

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

#         model = GAT(
#             x.shape[1],
#             best_params["hidden_dim"],
#             best_params["dropout"],
#             best_params["heads"]
#         ).to(device)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=best_params["lr"],
#             weight_decay=best_params["weight_decay"]
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         best_loss = float("inf")
#         counter = 0

#         for epoch in range(MAX_EPOCHS):
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

#             if counter >= PATIENCE:
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

#     print("\n===== RESULTADOS GAT =====")
#     print(results)

#     plot_confusion_matrix(y_true, y_pred, "GAT")
#     plot_pr_curve(y_true, y_prob, "GAT")

#     return results


















# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F

# from torch_geometric.nn import GATConv
# from torch_geometric.utils import to_undirected, add_self_loops

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# # ==============================
# # MODEL
# # ==============================
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()

#         self.conv1 = GATConv(
#             in_channels,
#             hidden_channels,
#             heads=4,
#             dropout=0.6
#         )

#         self.conv2 = GATConv(
#             hidden_channels * 4,
#             hidden_channels,
#             heads=4,
#             dropout=0.6
#         )

#         self.conv3 = GATConv(
#             hidden_channels * 4,
#             1,
#             heads=1,
#             concat=False
#         )

#         self.dropout = 0.6

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)

#         return x.squeeze()


# # ==============================
# # UTILS
# # ==============================
# def compute_class_weight(y):
#     pos = (y == 1).sum().item()
#     neg = (y == 0).sum().item()

#     if pos == 0:
#         return torch.tensor(1.0)

#     return torch.tensor(neg / pos)


# def find_best_threshold(y_true, y_prob):
#     thresholds = np.linspace(0.1, 0.9, 50)

#     best_f1 = 0
#     best_t = 0.5

#     for t in thresholds:
#         y_pred = (y_prob >= t).astype(int)

#         if len(np.unique(y_pred)) < 2:
#             continue

#         f1 = f1_score(y_true, y_pred)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_t = t

#     return best_t


# # ==============================
# # GRAPH BUILDER
# # ==============================
# def build_graph(df, edges):

#     feature_cols = [
#         c for c in df.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     x = torch.tensor(df[feature_cols].values, dtype=torch.float)
#     y = torch.tensor(df["class"].values, dtype=torch.float)

#     node_map = {tx: i for i, tx in enumerate(df["txId"].values)}

#     edge_list = []

#     for _, row in edges.iterrows():
#         if row["txId1"] in node_map and row["txId2"] in node_map:
#             edge_list.append([
#                 node_map[row["txId1"]],
#                 node_map[row["txId2"]]
#             ])

#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

#     edge_index = to_undirected(edge_index)
#     edge_index, _ = add_self_loops(edge_index)

#     return x, y, edge_index


# # ==============================
# # MAIN
# # ==============================
# def run_gat_temporal(df_train, df_val, df_test, edges):
#     print("\n===== GAT TEMPORAL (WEBER PAPER STYLE) =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     # ======================
#     # NORMALIZAÇÃO (SEM LEAKAGE)
#     # ======================
#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     scaler = StandardScaler()
#     scaler.fit(df_train[feature_cols])

#     df_all[feature_cols] = scaler.transform(df_all[feature_cols])

#     df_all["time_step"] = df_all["time_step"].astype(int)

#     max_time = int(df_all["time_step"].max())

#     model = None

#     all_probs = []
#     all_true = []

#     # ======================
#     # LOOP TEMPORAL
#     # ======================
#     for t in range(30, max_time):

#         print(f"\n--- Time {t} → Predict {t+1} ---")

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         # grafo até t
#         x, y, edge_index = build_graph(df_past, edges)

#         train_mask = torch.tensor(
#             (df_past["time_step"] < t).values
#         )

#         # init modelo
#         if model is None:
#             model = GAT(x.shape[1], 32)

#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=0.003,
#             weight_decay=5e-4
#         )

#         pos_weight = compute_class_weight(y[train_mask])
#         criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         # ======================
#         # TREINO
#         # ======================
#         model.train()
#         for epoch in range(20):
#             logits = model(x, edge_index)

#             loss = criterion(
#                 logits[train_mask],
#                 y[train_mask]
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # ======================
#         # INFERÊNCIA (t+1)
#         # ======================
#         df_eval = df_all[df_all["time_step"] <= t + 1]

#         x_eval, y_eval, edge_eval = build_graph(df_eval, edges)

#         model.eval()
#         with torch.no_grad():
#             logits = model(x_eval, edge_eval)
#             probs = torch.sigmoid(logits).cpu().numpy()

#         idx_target = df_eval["time_step"] == (t + 1)

#         y_true = y_eval[idx_target].cpu().numpy()
#         y_prob = probs[idx_target]

#         if len(y_true) == 0:
#             continue

#         print(f"Amostras válidas: {len(y_true)}")

#         all_probs.extend(y_prob)
#         all_true.extend(y_true)

#     # ======================
#     # MÉTRICAS FINAIS
#     # ======================
#     if len(all_true) == 0:
#         print("\nERRO: nenhum dado válido para avaliação")
#         return None

#     all_probs = np.array(all_probs)
#     all_true = np.array(all_true)

#     best_t = find_best_threshold(all_true, all_probs)
#     y_pred = (all_probs >= best_t).astype(int)

#     results = {
#         "PR_AUC": average_precision_score(all_true, all_probs),
#         "F1": f1_score(all_true, y_pred),
#         "Precision": precision_score(all_true, y_pred),
#         "Recall": recall_score(all_true, y_pred),
#         "model": "GAT_TEMPORAL"
#     }

#     print("\n===== RESULTADOS GAT TEMPORAL =====")
#     print(results)

#     return results
