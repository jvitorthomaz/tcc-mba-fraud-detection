"""
Geração de features de grafo sem vazamento temporal.

Inclui:
- Grau do nó
- Média dos vizinhos (1-hop)
- Média dos vizinhos dos vizinhos (2-hop)
"""

import pandas as pd
import numpy as np


def add_graph_features(df, edges):
    print("\n===== GERANDO FEATURES DE GRAFO (SEM LEAKAGE) =====")

    df = df.copy().reset_index(drop=True)

    id_map = {tx: i for i, tx in enumerate(df["txId"].values)}
    n = len(df)

    feature_cols = [c for c in df.columns if c not in ["txId", "class", "time_step"]]
    X = df[feature_cols].values

    times = df["time_step"].values

    neighbors = [[] for _ in range(n)]

    # Filtra apenas arestas válidas
    valid_edges = edges[
        edges["txId1"].isin(id_map) & edges["txId2"].isin(id_map)
    ]

    # Construção do grafo respeitando tempo
    for _, row in valid_edges.iterrows():
        i = id_map[row["txId1"]]
        j = id_map[row["txId2"]]

        if times[j] <= times[i]:
            neighbors[i].append(j)
        if times[i] <= times[j]:
            neighbors[j].append(i)

    # ==============================
    # DEGREE
    # ==============================
    degree = np.array([len(neigh) for neigh in neighbors]).reshape(-1, 1)

    # ==============================
    # 1-HOP NEIGHBOR MEAN
    # ==============================
    neighbor_mean = np.zeros_like(X)

    for i in range(n):
        if neighbors[i]:
            neighbor_mean[i] = X[neighbors[i]].mean(axis=0)

    # ==============================
    # 2-HOP NEIGHBOR MEAN
    # ==============================
    neighbor_2hop_mean = np.zeros_like(X)

    for i in range(n):
        second_hop = []

        for j in neighbors[i]:
            second_hop.extend(neighbors[j])

        second_hop = list(set(second_hop) - {i})

        if second_hop:
            neighbor_2hop_mean[i] = X[second_hop].mean(axis=0)

    # ==============================
    # CONCATENA FEATURES
    # ==============================
    df_degree = pd.DataFrame(degree, columns=["degree"])

    df_neighbor_mean = pd.DataFrame(
        neighbor_mean,
        columns=[f"neighbor_mean_{col}" for col in feature_cols]
    )

    df_neighbor_2hop = pd.DataFrame(
        neighbor_2hop_mean,
        columns=[f"neighbor_2hop_{col}" for col in feature_cols]
    )

    df_final = pd.concat(
        [df, df_degree, df_neighbor_mean, df_neighbor_2hop],
        axis=1
    )

    print("Features de grafo adicionadas (sem leakage)")

    return df_final
