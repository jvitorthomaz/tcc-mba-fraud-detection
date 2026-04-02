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

    valid_edges = edges[
        edges["txId1"].isin(id_map) & edges["txId2"].isin(id_map)
    ]

    for _, row in valid_edges.iterrows():
        i = id_map[row["txId1"]]
        j = id_map[row["txId2"]]

        # FILTRO TEMPORAL (ESSENCIAL)
        if times[j] <= times[i]:
            neighbors[i].append(j)
        if times[i] <= times[j]:
            neighbors[j].append(i)

    # Grau
    degree = np.array([len(neigh) for neigh in neighbors]).reshape(-1, 1)

    # Média dos vizinhos
    neighbor_mean = np.zeros_like(X)

    for i in range(n):
        if neighbors[i]:
            neighbor_mean[i] = X[neighbors[i]].mean(axis=0)
        else:
            neighbor_mean[i] = 0

    df_degree = pd.DataFrame(degree, columns=["degree"])

    df_neighbor_mean = pd.DataFrame(
        neighbor_mean,
        columns=[f"neighbor_mean_{col}" for col in feature_cols]
    )

    df_final = pd.concat([df, df_degree, df_neighbor_mean], axis=1)

    print("Features de grafo adicionadas (sem leakage)!")

    return df_final

# import pandas as pd
# import numpy as np


# def add_graph_features(df, edges):
#     """
#     Adiciona features de vizinhança ao dataset:
#     - Grau do nó
#     - Média das features dos vizinhos

#     Parâmetros:
#         df: DataFrame com txId e features
#         edges: DataFrame com colunas [txId1, txId2]

#     Retorna:
#         df com novas features
#     """

#     print("\n===== GERANDO FEATURES DE GRAFO =====")

#     df = df.copy().reset_index(drop=True)

#     # ==============================
#     # MAPEAR IDS
#     # ==============================

#     id_map = {tx: i for i, tx in enumerate(df["txId"].values)}
#     n = len(df)

#     # ==============================
#     # MATRIZ DE FEATURES
#     # ==============================

#     feature_cols = [c for c in df.columns if c not in ["txId", "class"]]
#     X = df[feature_cols].values

#     # ==============================
#     # CRIAR LISTA DE VIZINHOS
#     # ==============================

#     neighbors = [[] for _ in range(n)]

#     valid_edges = edges[
#         edges["txId1"].isin(id_map) & edges["txId2"].isin(id_map)
#     ]

#     for _, row in valid_edges.iterrows():
#         i = id_map[row["txId1"]]
#         j = id_map[row["txId2"]]

#         neighbors[i].append(j)
#         neighbors[j].append(i)

#     # ==============================
#     # FEATURE: GRAU
#     # ==============================

#     degree = np.array([len(neigh) for neigh in neighbors]).reshape(-1, 1)

#     # ==============================
#     # FEATURE: MÉDIA DOS VIZINHOS
#     # ==============================

#     neighbor_mean = np.zeros_like(X)

#     for i in range(n):
#         if len(neighbors[i]) > 0:
#             neighbor_mean[i] = X[neighbors[i]].mean(axis=0)
#         else:
#             neighbor_mean[i] = 0

#     # ==============================
#     # CONVERTER PARA DATAFRAME (UMA VEZ)
#     # ==============================

#     df_degree = pd.DataFrame(degree, columns=["degree"])

#     df_neighbor_mean = pd.DataFrame(
#         neighbor_mean,
#         columns=[f"neighbor_mean_{col}" for col in feature_cols]
#     )

#     # ==============================
#     # CONCAT FINAL (SEM FRAGMENTAÇÃO)
#     # ==============================

#     df_final = pd.concat(
#         [df, df_degree, df_neighbor_mean],
#         axis=1
#     )

#     print("Features de grafo adicionadas!")

#     return df_final
