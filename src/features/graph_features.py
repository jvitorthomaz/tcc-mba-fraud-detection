import pandas as pd
import numpy as np


def add_graph_features(df: pd.DataFrame, edges: pd.DataFrame):

    print("\n===== GERANDO FEATURES DE GRAFO =====")

    df = df.copy()

    # ==============================
    # 1. GRAU
    # ==============================
    degree = pd.concat([edges["txId1"], edges["txId2"]]).value_counts()
    df["degree"] = df["txId"].map(degree).fillna(0)

    # ==============================
    # 2. IN / OUT DEGREE
    # ==============================
    in_degree = edges["txId2"].value_counts()
    out_degree = edges["txId1"].value_counts()

    df["in_degree"] = df["txId"].map(in_degree).fillna(0)
    df["out_degree"] = df["txId"].map(out_degree).fillna(0)

    # ==============================
    # 3. VIZINHANÇA
    # ==============================
    feature_cols = [
        col for col in df.columns
        if col not in ["txId", "class"]
    ]

    features_dict = df.set_index("txId")[feature_cols].to_dict(orient="index")

    neighbors = {}

    for _, row in edges.iterrows():
        a, b = row["txId1"], row["txId2"]

        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    neighbor_features = []

    for tx in df["txId"]:
        neighs = neighbors.get(tx, [])

        if len(neighs) == 0:
            neighbor_features.append([0] * len(feature_cols))
            continue

        vals = []

        for n in neighs:
            if n in features_dict:
                vals.append(list(features_dict[n].values()))

        if len(vals) == 0:
            neighbor_features.append([0] * len(feature_cols))
        else:
            neighbor_features.append(np.mean(vals, axis=0))

    neighbor_features = np.array(neighbor_features)

    for i, col in enumerate(feature_cols):
        df[f"neighbor_mean_{col}"] = neighbor_features[:, i]

    print("Features de grafo adicionadas!")

    return df
