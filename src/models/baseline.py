"""
Treinamento dos modelos baseline com:
- Optuna
- Threshold tuning
- Métricas completas

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
# AVALIAÇÃO (SEM LEAKAGE)
# ==============================
def evaluate(y_true, y_probs, threshold, model_name):
    preds = (y_probs >= threshold).astype(int)

    # ======================
    # PLOTS (TESTE)
    # ======================
    plot_confusion_matrix(y_true, preds, model_name)
    plot_pr_curve(y_true, y_probs, model_name)

    return {
        "PR_AUC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Threshold": threshold
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
    # THRESHOLD VIA VAL (CORRETO)
    # ==============================
    val_probs_xgb = model_xgb.predict_proba(X_val)[:, 1]
    val_probs_rf = model_rf.predict_proba(X_val)[:, 1]
    val_probs_lr = model_lr.predict_proba(X_val)[:, 1]

    t_xgb = find_best_threshold(y_val, val_probs_xgb)
    t_rf = find_best_threshold(y_val, val_probs_rf)
    t_lr = find_best_threshold(y_val, val_probs_lr)

    # ==============================
    # TESTE
    # ==============================
    probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
    probs_rf = model_rf.predict_proba(X_test)[:, 1]
    probs_lr = model_lr.predict_proba(X_test)[:, 1]

    probs_ensemble = (probs_xgb + probs_rf + probs_lr) / 3

    # ==============================
    # AVALIAÇÃO FINAL (SEM LEAKAGE)
    # ==============================
    results = {
        "XGBoost": evaluate(y_test, probs_xgb, t_xgb, "XGBoost"),
        "RandomForest": evaluate(y_test, probs_rf, t_rf, "RandomForest"),
        "LogisticRegression": evaluate(y_test, probs_lr, t_lr, "LogisticRegression"),
        "Ensemble": evaluate(y_test, probs_ensemble, 0.5, "Ensemble"),
    }

    # ==============================
    # OUTPUT FINAL
    # ==============================
    print("\n===== RESULTADOS FINAIS =====")
    results_df = pd.DataFrame(results).T
    print(results_df)

    results_df.to_csv("results/baseline_results.csv")

    return results_df

# -----------------------------------

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
from src.evaluation.plots_summary import plot_metric_comparison
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve
)

os.makedirs("results/figures", exist_ok=True)


# ==============================
# THRESHOLD OTIMIZADO
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
# AVALIAÇÃO GENÉRICA (SEM PLOT)
# ==============================
def evaluate_simple(y_true, y_probs, threshold):
    preds = (y_probs >= threshold).astype(int)

    return {
        "PR_AUC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
    }


# ==============================
# AVALIAÇÃO FINAL (COM PLOT)
# ==============================
def evaluate_test(y_true, y_probs, threshold, model_name):
    preds = (y_probs >= threshold).astype(int)

    plot_confusion_matrix(y_true, preds, model_name)
    plot_pr_curve(y_true, y_probs, model_name)

    return {
        "PR_AUC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Threshold": threshold
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

    # treino final
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    # ==============================
    # MODELOS
    # ==============================
    model_xgb = tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
    model_rf = tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
    model_lr = tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full)

    # ==============================
    # PROBABILIDADES
    # ==============================
    def get_probs(model):
        return (
            model.predict_proba(X_train)[:, 1],
            model.predict_proba(X_val)[:, 1],
            model.predict_proba(X_test)[:, 1]
        )

    probs_train_xgb, probs_val_xgb, probs_test_xgb = get_probs(model_xgb)
    probs_train_rf, probs_val_rf, probs_test_rf = get_probs(model_rf)
    probs_train_lr, probs_val_lr, probs_test_lr = get_probs(model_lr)

    # ==============================
    # THRESHOLD (VAL)
    # ==============================
    t_xgb = find_best_threshold(y_val, probs_val_xgb)
    t_rf = find_best_threshold(y_val, probs_val_rf)
    t_lr = find_best_threshold(y_val, probs_val_lr)

    # ==============================
    # AVALIAÇÃO
    # ==============================
    results = {}

    for name, y_p_train, y_p_val, y_p_test, t in [
        ("XGBoost", probs_train_xgb, probs_val_xgb, probs_test_xgb, t_xgb),
        ("RandomForest", probs_train_rf, probs_val_rf, probs_test_rf, t_rf),
        ("LogisticRegression", probs_train_lr, probs_val_lr, probs_test_lr, t_lr),
    ]:

        # results[f"{name}_Train"] = evaluate_simple(y_train, y_p_train, t)
        results[f"{name}_Train"] = {
            "PR_AUC": average_precision_score(y_train, y_p_train),
            "F1": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
        }
        results[f"{name}_Val"] = evaluate_simple(y_val, y_p_val, t)
        results[f"{name}_Test"] = evaluate_test(y_test, y_p_test, t, name)

    # ==============================
    # OUTPUT
    # ==============================
    results_df = pd.DataFrame(results).T

    print("\n===== RESULTADOS COMPLETOS =====")
    print(results_df)

    results_df.to_csv("results/baseline_results_detailed.csv")

    # ==============================
    # GRÁFICOS DE ARTIGO (NOVO)
    # ==============================
    plot_metric_comparison(results_df, "Recall", "recall_comparison")
    plot_metric_comparison(results_df, "Precision", "precision_comparison")
    plot_metric_comparison(results_df, "F1", "f1_comparison")
    plot_metric_comparison(results_df, "PR_AUC", "pr_auc_comparison")

    return results_df

# import os
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

# # plots
# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )


# # ==============================
# # GARANTE PASTA DE OUTPUT
# # ==============================
# os.makedirs("results/figures", exist_ok=True)


# # ==============================
# # THRESHOLD OTIMIZADO (F1)
# # ==============================
# def find_best_threshold(y_true, y_probs):
#     thresholds = np.linspace(0.1, 0.9, 50)

#     best_f1 = 0
#     best_t = 0.5

#     for t in thresholds:
#         preds = (y_probs >= t).astype(int)
#         f1 = f1_score(y_true, preds, zero_division=0)

#         if f1 > best_f1:
#             best_f1 = f1
#             best_t = t

#     return best_t


# # ==============================
# # AVALIAÇÃO COMPLETA + PLOTS
# # ==============================
# def evaluate(y_true, y_probs, model_name):
#     t = find_best_threshold(y_true, y_probs)
#     preds = (y_probs >= t).astype(int)

#     # ======================
#     # PLOTS (SALVA AUTOMATICAMENTE)
#     # ======================
#     plot_confusion_matrix(y_true, preds, model_name)
#     plot_pr_curve(y_true, y_probs, model_name)

#     return {
#         "PR_AUC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds, zero_division=0),
#         "Precision": precision_score(y_true, preds, zero_division=0),
#         "Recall": recall_score(y_true, preds, zero_division=0),
#         "Threshold": t
#     }


# # ==============================
# # MAIN
# # ==============================
# def run_baseline_models(df_train, df_val, df_test):
#     print("\n===== TREINANDO MODELOS BASELINE =====")

#     feature_cols = [
#         c for c in df_train.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     X_train = df_train[feature_cols]
#     y_train = df_train["class"]

#     X_val = df_val[feature_cols]
#     y_val = df_val["class"]

#     X_test = df_test[feature_cols]
#     y_test = df_test["class"]

#     # ==============================
#     # TREINO FINAL = TRAIN + VAL
#     # ==============================
#     X_train_full = pd.concat([X_train, X_val])
#     y_train_full = pd.concat([y_train, y_val])

#     # ==============================
#     # MODELOS (OPTUNA)
#     # ==============================
#     model_xgb = tune_xgboost(
#         X_train, y_train, X_val, y_val,
#         X_train_full, y_train_full
#     )

#     model_rf = tune_random_forest(
#         X_train, y_train, X_val, y_val,
#         X_train_full, y_train_full
#     )

#     model_lr = tune_logistic_regression(
#         X_train, y_train, X_val, y_val,
#         X_train_full, y_train_full
#     )

#     # ==============================
#     # PREDIÇÕES
#     # ==============================
#     probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
#     probs_rf = model_rf.predict_proba(X_test)[:, 1]
#     probs_lr = model_lr.predict_proba(X_test)[:, 1]

#     probs_ensemble = (probs_xgb + probs_rf + probs_lr) / 3

#     # ==============================
#     # AVALIAÇÃO + PLOTS
#     # ==============================
#     results = {
#         "XGBoost": evaluate(y_test, probs_xgb, "XGBoost"),
#         "RandomForest": evaluate(y_test, probs_rf, "RandomForest"),
#         "LogisticRegression": evaluate(y_test, probs_lr, "LogisticRegression"),
#         "Ensemble": evaluate(y_test, probs_ensemble, "Ensemble"),
#     }

#     # ==============================
#     # OUTPUT FINAL
#     # ==============================
#     print("\n===== RESULTADOS FINAIS =====")
#     results_df = pd.DataFrame(results).T
#     print(results_df)

#     # salva CSV (útil pro artigo)
#     results_df.to_csv("results/baseline_results.csv")

#     return results_df


