import os

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import optuna

from torch_geometric.nn import GCNConv
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
from src.evaluation.metrics import evaluate_timestep
from src.utils.config import RANDOM_SEED

os.makedirs("results/per_timestep", exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, 1)

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

        model = GCN(x.shape[1], hidden_dim, dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    return np.mean(scores) if scores else 0


def tune_gcn(df_train, df_val, edges):
    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)
    study = optuna.create_study(direction="maximize")

    #study.optimize(lambda trial: objective(trial, df_all, edges), n_trials=50)
    study.optimize(lambda trial: objective(trial, df_all, edges), n_trials=5)


    print("\n===== MELHORES PARÂMETROS GCN =====")
    print(study.best_params)

    return study.best_params


def run_gcn_temporal(df_train, df_val, df_test, edges):
    print("\n===== GCN =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    df_all[feature_cols] = scaler.transform(df_all[feature_cols])

    df_all["time_step"] = df_all["time_step"].astype(int)
    max_time = int(df_all["time_step"].max())

    best_params = tune_gcn(df_train, df_val, edges)

    all_probs, all_true, all_preds = [], [], []
    timestep_records = []

    for t in range(30, max_time):
        df_past = df_all[df_all["time_step"] <= t]
        df_target = df_all[df_all["time_step"] == t + 1]

        if len(df_target) == 0:
            continue

        x, y, edge_index, _ = build_graph(df_past, edges)
        if x is None:
            continue

        train_mask = torch.tensor((df_past["time_step"] < t).values, device=device)
        val_mask = torch.tensor((df_past["time_step"] == t).values, device=device)

        model = GCN(x.shape[1], best_params["hidden_dim"], best_params["dropout"]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

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

        with torch.no_grad():
            probs = torch.sigmoid(model(x_eval, edge_eval)).cpu().numpy()

        idx = (df_eval["time_step"] == (t + 1)).values

        y_step_true = y_eval[idx].cpu().numpy()
        y_step_prob = probs[idx]
        y_step_pred = (y_step_prob >= best_t).astype(int)

        all_true.extend(y_step_true)
        all_probs.extend(y_step_prob)
        all_preds.extend(y_step_pred)

        timestep_records.append({
            "time_step": t + 1,
            **evaluate_timestep(y_step_true, y_step_pred, y_step_prob)
        })

    pd.DataFrame(timestep_records).to_csv("results/per_timestep/GCN.csv", index=False)

    y_true = np.array(all_true)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    results = {
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }

    print(results)

    plot_confusion_matrix(y_true, y_pred, "GCN")
    plot_pr_curve(y_true, y_prob, "GCN")

    return results
