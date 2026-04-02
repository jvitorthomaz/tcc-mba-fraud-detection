import torch
from torch_geometric.data import Data


def build_graph_data(df_train, df_val, df_test, edges):
    df_all = df_train.copy()
    df_all = df_all.append(df_val)
    df_all = df_all.append(df_test)

    df_all = df_all.reset_index(drop=True)

    feature_cols = [col for col in df_all.columns if col not in ["class", "txId", "time"]]

    x = torch.tensor(df_all[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df_all["class"].values, dtype=torch.long)

    edge_index = torch.tensor(edges.T, dtype=torch.long)

    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)

    train_mask = torch.zeros(len(df_all), dtype=torch.bool)
    val_mask = torch.zeros(len(df_all), dtype=torch.bool)
    test_mask = torch.zeros(len(df_all), dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data
