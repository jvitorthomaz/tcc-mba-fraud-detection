import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mannwhitneyu, entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==============================
# 1. CARREGAMENTO
# ==============================

df = pd.read_csv("../datasets/transaction_dataset.csv")

df = df.drop(columns=["Unnamed: 0", "Index", "Address"], errors="ignore")
df["FLAG"] = df["FLAG"].astype(int)

numeric_cols = df.drop(columns=["FLAG"]).select_dtypes(include=np.number).columns

print("\n===== CARACTERIZAÇÃO =====")
print("Amostras:", df.shape[0])
print("Features:", len(numeric_cols))

class_dist = df["FLAG"].value_counts(normalize=True)
print("\nDistribuição de classes:")
print(class_dist)
print("Entropia do target:", entropy(class_dist))

# ==============================
# 2. VALORES FALTANTES
# ==============================

print("\n===== VALORES FALTANTES =====")
missing = df[numeric_cols].isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print(missing)
else:
    print("Sem valores faltantes.")

# ==============================
# 3. ESTATÍSTICAS UNIVARIADAS
# ==============================

stats = pd.DataFrame({
    "mean": df[numeric_cols].mean(),
    "std": df[numeric_cols].std(),
    "skew": df[numeric_cols].apply(skew),
    "kurtosis": df[numeric_cols].apply(kurtosis)
})

stats.to_csv("ethereum_univariate_stats.csv")

# ==============================
# 4. REMOVER FEATURES CONSTANTES
# ==============================

constant_features = [col for col in numeric_cols if df[col].nunique() <= 1]

if len(constant_features) > 0:
    print("\nRemovendo features constantes:")
    print(constant_features)
    df = df.drop(columns=constant_features)
    numeric_cols = df.drop(columns=["FLAG"]).select_dtypes(include=np.number).columns

# ==============================
# 5. SEPARABILIDADE ESTATÍSTICA
# ==============================

def cohens_d(x1, x2):
    std1 = x1.std()
    std2 = x2.std()
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0:
        return 0
    return (x1.mean() - x2.mean()) / pooled_std

results = []
effects = []

for col in numeric_cols:
    fraud = df[df["FLAG"] == 1][col].fillna(0)
    legit = df[df["FLAG"] == 0][col].fillna(0)

    try:
        stat, p = mannwhitneyu(fraud, legit, alternative='two-sided')
    except:
        p = 1

    results.append((col, p))
    effects.append((col, cohens_d(fraud, legit)))

pvalues = pd.DataFrame(results, columns=["feature", "p_value"])
effects_df = pd.DataFrame(effects, columns=["feature", "cohens_d"])

pvalues.sort_values("p_value").to_csv("ethereum_pvalues.csv", index=False)
effects_df.sort_values("cohens_d", key=abs, ascending=False).to_csv("ethereum_effect_sizes.csv", index=False)

# ==============================
# 6. INFORMAÇÃO MÚTUA
# ==============================

X_mi = df[numeric_cols].fillna(0)

mi = mutual_info_classif(X_mi, df["FLAG"], random_state=42)

mi_df = pd.DataFrame({
    "feature": numeric_cols,
    "mutual_information": mi
}).sort_values("mutual_information", ascending=False)

mi_df.to_csv("ethereum_mutual_information.csv", index=False)

# ==============================
# 7. MULTICOLINEARIDADE (VIF)
# ==============================

print("\n===== MULTICOLINEARIDADE =====")

X_vif = df[numeric_cols].fillna(0)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif_data.sort_values("VIF", ascending=False).to_csv("ethereum_vif.csv", index=False)

# ==============================
# 8. PCA
# ==============================

print("\n===== PCA =====")

X_scaled = StandardScaler().fit_transform(X_vif)

pca = PCA(n_components=5)
pca.fit(X_scaled)

print("Variância explicada pelos 5 primeiros componentes:")
print(pca.explained_variance_ratio_)

# ==============================
# 9. CORRELAÇÃO COM TARGET
# ==============================

corr = df.corr(numeric_only=True)["FLAG"].sort_values(ascending=False)
corr.to_csv("ethereum_correlations.csv")

print("\nEDA Ethereum concluída com sucesso.")