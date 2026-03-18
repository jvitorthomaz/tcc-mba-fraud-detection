import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. CARREGAMENTO
# ==============================

features = pd.read_csv("datasets/elliptic_txs_features.csv", header=None)
classes = pd.read_csv("datasets/elliptic_txs_classes.csv")
edges = pd.read_csv("datasets/elliptic_txs_edgelist.csv")

features = features.rename(columns={0: "txId"})
df = features.merge(classes, on="txId", how="left")

# Mantém unknown
df["class"] = df["class"].astype(str)

numeric_cols = df.drop(columns=["txId", "class"]).columns

# ==============================
# 2. CONSTRUÇÃO DO GRAFO
# ==============================

G = nx.from_pandas_edgelist(
    edges, source="txId1", target="txId2", create_using=nx.DiGraph()
)

# adicionar atributo class aos nós (incluindo unknown)
class_dict = df.set_index("txId")["class"].to_dict()
nx.set_node_attributes(G, class_dict, "class")

# ==============================
# 3. ASSORTATIVIDADE
# ==============================

assortativity = nx.attribute_assortativity_coefficient(G, "class")
print("\nAssortatividade por classe:", assortativity)

# ==============================
# 4. CLUSTERING COEFFICIENT POR CLASSE
# ==============================

clustering = nx.clustering(G.to_undirected())
df["clustering"] = df["txId"].map(clustering)

print("\nClustering médio por classe:")
print(df.groupby("class")["clustering"].mean())

# ==============================
# 5. DISTRIBUIÇÃO DE GRAU POR CLASSE
# ==============================

degrees = dict(G.degree())
df["degree"] = df["txId"].map(degrees)

plt.figure()
sns.histplot(data=df, x="degree", hue="class", bins=50, kde=True)
plt.title("Distribuição de Grau por Classe")
plt.savefig("elliptic_degree_distribution.png")
plt.close()

# ==============================
# 6. CORRELAÇÃO (Pearson + Spearman)
# ==============================

pearson_corr = df[numeric_cols].corr(method="pearson")
spearman_corr = df[numeric_cols].corr(method="spearman")

pearson_corr.to_csv("elliptic_pearson_corr.csv")
spearman_corr.to_csv("elliptic_spearman_corr.csv")

# ==============================
# 7. KOLMOGOROV-SMIRNOV (3 comparações)
# ==============================

ks_results = []

classes_unique = df["class"].unique()

for col in numeric_cols:
    for i in range(len(classes_unique)):
        for j in range(i+1, len(classes_unique)):
            c1 = classes_unique[i]
            c2 = classes_unique[j]

            group1 = df[df["class"] == c1][col]
            group2 = df[df["class"] == c2][col]

            stat, p = ks_2samp(group1, group2)
            ks_results.append((col, c1, c2, stat, p))

ks_df = pd.DataFrame(
    ks_results,
    columns=["feature", "class_1", "class_2", "ks_stat", "p_value"]
)

ks_df.sort_values("ks_stat", ascending=False).to_csv(
    "elliptic_ks_test_all_classes.csv", index=False
)

# ==============================
# 8. MUTUAL INFORMATION (multiclasse)
# ==============================

le = LabelEncoder()
y_encoded = le.fit_transform(df["class"])

mi = mutual_info_classif(df[numeric_cols], y_encoded)

mi_df = pd.DataFrame({"feature": numeric_cols, "mi": mi})
top_features = mi_df.sort_values("mi", ascending=False)["feature"].head(5)

for col in top_features:
    plt.figure()
    sns.boxplot(x="class", y=col, data=df)
    plt.title(f"Boxplot - {col}")
    plt.savefig(f"elliptic_boxplot_{col}.png")
    plt.close()

print("\nEDA estrutural Elliptic concluída.")