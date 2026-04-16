"""
Treinamento dos modelos baseline com:
- Optuna
- Threshold tuning
- Métricas completas
"""

import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from src.models.optuna_xgboost import tune_xgboost
from src.models.optuna_random_forest import tune_random_forest
from src.models.optuna_logistic_regression import tune_logistic_regression

# plots
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve
)


# ==============================
# GARANTE PASTA DE OUTPUT
# ==============================
os.makedirs("results/figures", exist_ok=True)


# ==============================
# THRESHOLD OTIMIZADO (F1)
# ==============================
def find_best_threshold(y_true, y_probs):
    thresholds = np.linspace(0.1, 0.9, 50)

    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t


# ==============================
# AVALIAÇÃO COMPLETA + PLOTS
# ==============================
def evaluate(y_true, y_probs, model_name):
    t = find_best_threshold(y_true, y_probs)
    preds = (y_probs >= t).astype(int)

    # ======================
    # PLOTS (SALVA AUTOMATICAMENTE)
    # ======================
    plot_confusion_matrix(y_true, preds, model_name)
    plot_pr_curve(y_true, y_probs, model_name)

    return {
        "PR_AUC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Threshold": t
    }


# ==============================
# MAIN
# ==============================
def run_baseline_models(df_train, df_val, df_test):
    print("\n===== TREINANDO MODELOS BASELINE =====")

    feature_cols = [
        c for c in df_train.columns
        if c not in ["txId", "class", "time_step"]
    ]

    X_train = df_train[feature_cols]
    y_train = df_train["class"]

    X_val = df_val[feature_cols]
    y_val = df_val["class"]

    X_test = df_test[feature_cols]
    y_test = df_test["class"]

    # ==============================
    # TREINO FINAL = TRAIN + VAL
    # ==============================
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    # ==============================
    # MODELOS (OPTUNA)
    # ==============================
    model_xgb = tune_xgboost(
        X_train, y_train, X_val, y_val,
        X_train_full, y_train_full
    )

    model_rf = tune_random_forest(
        X_train, y_train, X_val, y_val,
        X_train_full, y_train_full
    )

    model_lr = tune_logistic_regression(
        X_train, y_train, X_val, y_val,
        X_train_full, y_train_full
    )

    # ==============================
    # PREDIÇÕES
    # ==============================
    probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
    probs_rf = model_rf.predict_proba(X_test)[:, 1]
    probs_lr = model_lr.predict_proba(X_test)[:, 1]

    probs_ensemble = (probs_xgb + probs_rf + probs_lr) / 3

    # ==============================
    # AVALIAÇÃO + PLOTS
    # ==============================
    results = {
        "XGBoost": evaluate(y_test, probs_xgb, "XGBoost"),
        "RandomForest": evaluate(y_test, probs_rf, "RandomForest"),
        "LogisticRegression": evaluate(y_test, probs_lr, "LogisticRegression"),
        "Ensemble": evaluate(y_test, probs_ensemble, "Ensemble"),
    }

    # ==============================
    # OUTPUT FINAL
    # ==============================
    print("\n===== RESULTADOS FINAIS =====")
    results_df = pd.DataFrame(results).T
    print(results_df)

    # salva CSV (útil pro artigo)
    results_df.to_csv("results/baseline_results.csv")

    return results_df




# import pandas as pd
# import numpy as np

# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
#     average_precision_score
# )

# from src.models.optuna_xgboost import tune_xgboost
# from src.models.optuna_random_forest import tune_random_forest
# from src.models.optuna_logistic_regression import tune_logistic_regression


# def find_best_threshold(y_true, y_probs):
#     thresholds = np.linspace(0.1, 0.9, 50)

#     best_f1 = 0
#     best_t = 0.5

#     for t in thresholds:
#         preds = (y_probs >= t).astype(int)
#         f1 = f1_score(y_true, preds)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_t = t

#     return best_t


# def evaluate(y_true, y_probs):
#     t = find_best_threshold(y_true, y_probs)
#     preds = (y_probs >= t).astype(int)

#     return {
#         "PR_AUC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds),
#         "Precision": precision_score(y_true, preds),
#         "Recall": recall_score(y_true, preds)
#     }


# def run_baseline_models(df_train, df_val, df_test):
#     print("\n===== TREINANDO MODELOS BASELINE =====")

#     feature_cols = [c for c in df_train.columns if c not in ["txId", "class", "time_step"]]

#     X_train = df_train[feature_cols]
#     y_train = df_train["class"]

#     X_val = df_val[feature_cols]
#     y_val = df_val["class"]

#     X_test = df_test[feature_cols]
#     y_test = df_test["class"]

#     # JUNTA TRAIN + VAL (melhor prática)
#     X_train_full = pd.concat([X_train, X_val])
#     y_train_full = pd.concat([y_train, y_val])

#     # ==============================
#     # OPTUNA MODELS (CORRIGIDOS)
#     # ==============================

#     model_xgb = tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
#     model_rf = tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
#     model_lr = tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full)

#     # ==============================
#     # PREDIÇÕES
#     # ==============================

#     probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
#     probs_rf = model_rf.predict_proba(X_test)[:, 1]
#     probs_lr = model_lr.predict_proba(X_test)[:, 1]

#     probs_ensemble = (probs_xgb + probs_rf + probs_lr) / 3

#     # ==============================
#     # AVALIAÇÃO
#     # ==============================

#     results = {
#         "XGBoost": evaluate(y_test, probs_xgb),
#         "RandomForest": evaluate(y_test, probs_rf),
#         "LogisticRegression": evaluate(y_test, probs_lr),
#         "Ensemble": evaluate(y_test, probs_ensemble),
#     }


#     print("\n===== RESULTADOS FINAIS =====")
#     print(pd.DataFrame(results).T)







# import pandas as pd
# import numpy as np

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
# from sklearn.utils.class_weight import compute_class_weight

# from src.models.utils import find_best_threshold




# def run_baseline_models(df_train, df_val, df_test, use_timestep=False):

#     target = "class"

#     # =========================
#     # DEFINIÇÃO DE FEATURES
#     # =========================
#     cols_to_drop = ["class", "txId"]

#     if not use_timestep:
#         cols_to_drop.append("time_step")

#     X_train = df_train.drop(columns=cols_to_drop)
#     y_train = df_train[target]

#     X_val = df_val.drop(columns=cols_to_drop)
#     y_val = df_val[target]

#     X_test = df_test.drop(columns=cols_to_drop)
#     y_test = df_test[target]

#     # Garantir nomes válidos
#     X_train.columns = X_train.columns.astype(str)
#     X_val.columns = X_val.columns.astype(str)
#     X_test.columns = X_test.columns.astype(str)

#     # =========================
#     # CLASS IMBALANCE
#     # =========================
#     classes = np.unique(y_train)

#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=classes,
#         y=y_train
#     )

#     class_weight_dict = dict(zip(classes, weights))
#     scale_pos_weight = weights[1] / weights[0]

#     results = []

#     # =========================
#     # MODELO 1: LOGISTIC REGRESSION
#     # =========================
#     print("Treinando Logistic Regression...")

#     scaler = StandardScaler()

#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)

#     lr = LogisticRegression(
#         max_iter=2000,
#         class_weight=class_weight_dict,
#         solver="liblinear",
#         penalty="l2"
#     )

#     lr.fit(X_train_scaled, y_train)

#     y_val_prob = lr.predict_proba(X_val_scaled)[:, 1]
#     best_thresh = find_best_threshold(y_val, y_val_prob)

#     y_prob = lr.predict_proba(X_test_scaled)[:, 1]
#     y_pred = (y_prob >= best_thresh).astype(int)

#     results.append(_evaluate("LogisticRegression", y_test, y_prob, y_pred))

#     # =========================
#     # MODELO 2: RANDOM FOREST (MENOS RESTRITO)
#     # =========================
#     print("Treinando Random Forest...")

#     rf = RandomForestClassifier(
#         n_estimators=600,
#         max_depth=None,
#         min_samples_leaf=1,
#         max_features="sqrt",
#         class_weight=class_weight_dict,
#         n_jobs=-1,
#         random_state=42
#     )

#     rf.fit(X_train, y_train)

#     y_val_prob = rf.predict_proba(X_val)[:, 1]
#     best_thresh = find_best_threshold(y_val, y_val_prob)

#     y_prob = rf.predict_proba(X_test)[:, 1]
#     y_pred = (y_prob >= best_thresh).astype(int)

#     results.append(_evaluate("RandomForest", y_test, y_prob, y_pred))

#     # =========================
#     # MODELO 3: XGBOOST (CORRIGIDO)
#     # =========================
#     print("Treinando XGBoost...")

#     xgb = XGBClassifier(
#         n_estimators=600,
#         learning_rate=0.03,
#         max_depth=6,
#         min_child_weight=1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         gamma=0,
#         reg_alpha=0,
#         reg_lambda=1,
#         scale_pos_weight=scale_pos_weight,
#         eval_metric="logloss",
#         random_state=42
#     )

#     xgb.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=50,
#         verbose=False
#     )

#     y_val_prob = xgb.predict_proba(X_val)[:, 1]
#     best_thresh = find_best_threshold(y_val, y_val_prob)

#     y_prob = xgb.predict_proba(X_test)[:, 1]
#     y_pred = (y_prob >= best_thresh).astype(int)

#     results.append(_evaluate("XGBoost", y_test, y_prob, y_pred))

#     # =========================
#     # ENSEMBLE (SÓ RF + XGB)
#     # =========================
#     print("Calculando Ensemble (RF + XGB)...")

#     prob_rf = rf.predict_proba(X_test)[:, 1]
#     prob_xgb = xgb.predict_proba(X_test)[:, 1]

#     # Peso maior para XGBoost
#     ensemble_prob = 0.4 * prob_rf + 0.6 * prob_xgb

#     best_thresh = find_best_threshold(y_test, ensemble_prob)
#     ensemble_pred = (ensemble_prob >= best_thresh).astype(int)

#     results.append(_evaluate("Ensemble_RF_XGB", y_test, ensemble_prob, ensemble_pred))

#     # =========================
#     # RESULTADOS
#     # =========================
#     df_results = pd.DataFrame(results)

#     print("\n===== RESULTADOS FINAIS =====")
#     print(df_results)

#     df_results.to_csv("results/new_tables/baseline_results_final.csv", index=False)

#     return df_results


# def _evaluate(name, y_true, y_prob, y_pred):
#     return {
#         "model": name,
#         "PR_AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred),
#         "Precision": precision_score(y_true, y_pred),
#         "Recall": recall_score(y_true, y_pred)
#     }
