import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif

# ==============================
# 1. CARREGAMENTO
# ==============================

df = pd.read_csv("../datasets/transaction_dataset.csv")

df = df.drop(columns=["Unnamed: 0", "Index", "Address"], errors="ignore")
df["FLAG"] = df["FLAG"].astype(int)

numeric_cols = df.drop(columns=["FLAG"]).select_dtypes(include=np.number).columns

# ==============================
# 2. CORRELAÇÃO
# ==============================

pearson_corr = df[numeric_cols].corr(method="pearson")
spearman_corr = df[numeric_cols].corr(method="spearman")

pearson_corr.to_csv("ethereum_pearson_corr.csv")
spearman_corr.to_csv("ethereum_spearman_corr.csv")

# ==============================
# 3. KOLMOGOROV-SMIRNOV
# ==============================

ks_results = []

for col in numeric_cols:
    fraud = df[df["FLAG"] == 1][col].fillna(0)
    legit = df[df["FLAG"] == 0][col].fillna(0)
    stat, p = ks_2samp(fraud, legit)
    ks_results.append((col, stat, p))

ks_df = pd.DataFrame(ks_results, columns=["feature", "ks_stat", "p_value"])
ks_df.sort_values("ks_stat", ascending=False).to_csv(
    "ethereum_ks_test.csv", index=False
)

# ==============================
# 4. BOX-PLOTS TOP FEATURES (Mutual Info)
# ==============================

X_mi = df[numeric_cols].fillna(0)
mi = mutual_info_classif(X_mi, df["FLAG"], random_state=42)

mi_df = pd.DataFrame({"feature": numeric_cols, "mi": mi})
top_features = mi_df.sort_values("mi", ascending=False)["feature"].head(5)

for col in top_features:
    plt.figure()
    sns.boxplot(x="FLAG", y=col, data=df)
    plt.title(f"Boxplot - {col}")
    plt.savefig(f"ethereum_boxplot_{col}.png")
    plt.close()

# ==============================
# 5. HEATMAP TOP 20 FEATURES
# ==============================

top20 = mi_df.sort_values("mi", ascending=False)["feature"].head(20)
plt.figure(figsize=(12, 10))
sns.heatmap(df[top20].corr(), cmap="coolwarm", center=0)
plt.title("Heatmap Correlação - Top 20 Features")
plt.savefig("ethereum_heatmap_top20.png")
plt.close()

print("\nEDA avançada Ethereum concluída.")