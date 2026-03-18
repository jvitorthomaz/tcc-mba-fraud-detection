import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import zscore, skew, kurtosis

# ==============================
# 1. CARREGAMENTO
# ==============================

features = pd.read_csv("datasets/elliptic_txs_features.csv", header=None)
edges = pd.read_csv("datasets/elliptic_txs_edgelist.csv")
classes = pd.read_csv("datasets/elliptic_txs_classes.csv")

features.rename(columns={0: "txId"}, inplace=True)

df = features.merge(classes, on="txId", how="left")

# Mantém unknown
df["class"] = df["class"].astype(str)

# ==============================
# 2. ESTATÍSTICAS GERAIS
# ==============================

print("\n===== ESTATÍSTICAS GERAIS =====")
print("Número total de nós:", df.shape[0])
print("Número de features:", df.shape[1] - 2)

print("\nDistribuição de classes (%):")
print(df["class"].value_counts(normalize=True) * 100)

# ==============================
# 3. OUTLIERS (somente features)
# ==============================

numeric_cols = df.drop(columns=["txId", "class"]).columns

# Z-score
z_scores = np.abs(zscore(df[numeric_cols], nan_policy='omit'))
outliers_z = (z_scores > 3).sum().sum()
total_values = df[numeric_cols].shape[0] * df[numeric_cols].shape[1]
outlier_z_pct = (outliers_z / total_values) * 100

# IQR
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum().sum()

outlier_iqr_pct = (outliers_iqr / total_values) * 100

print(f"\nOutliers Z-score (%): {outlier_z_pct:.2f}")
print(f"Outliers IQR (%): {outlier_iqr_pct:.2f}")

# ==============================
# 4. CONSTRUÇÃO DO GRAFO
# ==============================

G = nx.from_pandas_edgelist(edges, source="txId1", target="txId2")

print("\n===== MÉTRICAS DE GRAFO =====")
print("Número de nós no grafo:", G.number_of_nodes())
print("Número de arestas:", G.number_of_edges())
print("Densidade:", nx.density(G))
print("Componentes conectados:", nx.number_connected_components(G))

degrees = dict(G.degree())
degree_values = list(degrees.values())

print("Grau médio:", np.mean(degree_values))
print("Grau máximo:", np.max(degree_values))
print("Desvio padrão do grau:", np.std(degree_values))

# ==============================
# 5. CENTRALIDADES
# ==============================

print("\nCalculando centralidades...")

degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G, k=500)
closeness = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)

centralities = pd.DataFrame({
    "txId": list(degree_centrality.keys()),
    "degree_centrality": list(degree_centrality.values()),
    "betweenness": list(betweenness.values()),
    "closeness": list(closeness.values()),
    "pagerank": list(pagerank.values())
})

df = df.merge(centralities, on="txId", how="left")

# ==============================
# 6. MÉDIAS POR CLASSE (incluindo unknown)
# ==============================

print("\n===== CENTRALIDADES MÉDIAS POR CLASSE =====")
print(df.groupby("class")[[
    "degree_centrality",
    "betweenness",
    "closeness",
    "pagerank"
]].mean())

# ==============================
# 7. ESTATÍSTICAS DAS FEATURES
# ==============================

stats = pd.DataFrame({
    "mean": df[numeric_cols].mean(),
    "std": df[numeric_cols].std(),
    "skew": df[numeric_cols].apply(skew),
    "kurtosis": df[numeric_cols].apply(kurtosis)
})

stats.to_csv("elliptic_feature_statistics.csv")

print("\nAnálise concluída e exportada.")