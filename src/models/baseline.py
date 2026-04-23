"""
Treinamento dos modelos baseline com:
- Optuna
- Threshold tuning
- Métricas completas

"""
# =============== threshold dinâmico temporal ==========================
import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from xgboost import XGBClassifier

from src.models.optuna_xgboost import tune_xgboost
from src.models.optuna_random_forest import tune_random_forest
from src.models.optuna_logistic_regression import tune_logistic_regression

from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve
)

os.makedirs("results/figures", exist_ok=True)


# ==============================
# THRESHOLD DINÂMICO
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
# TUNING GLOBAL (UMA VEZ)
# ==============================
def tune_global(df_train, df_val):

    feature_cols = [
        c for c in df_train.columns
        if c not in ["txId", "class", "time_step"]
    ]

    X_train = df_train[feature_cols]
    y_train = df_train["class"]

    X_val = df_val[feature_cols]
    y_val = df_val["class"]

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    print("\n===== TUNING GLOBAL =====")

    model_xgb = tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
    model_rf = tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
    model_lr = tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full)

    return {
        "XGBoost": model_xgb,
        "RandomForest": model_rf,
        "LogisticRegression": model_lr
    }


# ==============================
# MAIN TEMPORAL
# ==============================
def run_baseline_models(df_train, df_val, df_test):

    print("\n===== BASELINES TEMPORAIS =====")

    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    feature_cols = [
        c for c in df_all.columns
        if c not in ["txId", "class", "time_step"]
    ]

    df_all["time_step"] = df_all["time_step"].astype(int)
    max_time = int(df_all["time_step"].max())

    # ==============================
    # TUNING GLOBAL (fixo)
    # ==============================
    df_train_init = df_all[df_all["time_step"] < 32]
    df_val_init = df_all[df_all["time_step"] == 32]

    best_models = tune_global(df_train_init, df_val_init)

    # ==============================
    # STORAGE GLOBAL
    # ==============================
    all_probs = {name: [] for name in best_models.keys()}
    all_preds = {name: [] for name in best_models.keys()}
    all_true = []

    # ==============================
    # LOOP TEMPORAL
    # ==============================
    for t in range(30, max_time):

        df_past = df_all[df_all["time_step"] <= t]
        df_target = df_all[df_all["time_step"] == t + 1]

        if len(df_target) == 0:
            continue

        df_train_t = df_past[df_past["time_step"] < t]
        df_val_t = df_past[df_past["time_step"] == t]

        if len(df_train_t) == 0 or len(df_val_t) == 0:
            continue

        X_train = df_train_t[feature_cols]
        y_train = df_train_t["class"]

        X_val = df_val_t[feature_cols]
        y_val = df_val_t["class"]

        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])

        # ==============================
        # MODELOS (SEM VAZAMENTO)
        # ==============================
        models = {
            "XGBoost": XGBClassifier(**best_models["XGBoost"].get_params()),
            "RandomForest": clone(best_models["RandomForest"]),
            "LogisticRegression": clone(best_models["LogisticRegression"])
        }

        # ==============================
        # TREINO
        # ==============================
        for model in models.values():
            model.fit(X_train_full, y_train_full)

        X_target = df_target[feature_cols]
        y_target = df_target["class"].values

        # ==============================
        # POR MODELO
        # ==============================
        for name, model in models.items():

            probs_val = model.predict_proba(X_val)[:, 1]

            if len(np.unique(y_val)) >= 2:
                best_t = find_best_threshold(y_val, probs_val)
            else:
                best_t = 0.5

            probs_test = model.predict_proba(X_target)[:, 1]
            preds_test = (probs_test >= best_t).astype(int)

            all_probs[name].extend(probs_test)
            all_preds[name].extend(preds_test)

        all_true.extend(y_target)

    y_true = np.array(all_true)

    # ==============================
    # RESULTADOS FINAIS
    # ==============================
    results = {}

    for name in all_probs.keys():

        y_prob = np.array(all_probs[name])
        y_pred = np.array(all_preds[name])

        results[name] = {
            "PR_AUC": average_precision_score(y_true, y_prob),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
        }

        plot_confusion_matrix(y_true, y_pred, name)
        plot_pr_curve(y_true, y_prob, name)

    results_df = pd.DataFrame(results).T

    print("\n===== RESULTADOS FINAIS =====")
    print(results_df)

    results_df.to_csv("results/baseline_temporal_results.csv")

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

# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# from src.models.optuna_xgboost import tune_xgboost
# from src.models.optuna_random_forest import tune_random_forest
# from src.models.optuna_logistic_regression import tune_logistic_regression

# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# os.makedirs("results/figures", exist_ok=True)


# # ==============================
# # THRESHOLD DINÂMICO
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
# # TUNING GLOBAL (UMA VEZ)
# # ==============================
# def tune_global(df_train, df_val):

#     feature_cols = [
#         c for c in df_train.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     X_train = df_train[feature_cols]
#     y_train = df_train["class"]

#     X_val = df_val[feature_cols]
#     y_val = df_val["class"]

#     X_train_full = pd.concat([X_train, X_val])
#     y_train_full = pd.concat([y_train, y_val])

#     print("\n===== TUNING GLOBAL =====")

#     best_xgb = tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
#     best_rf = tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
#     best_lr = tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full)

#     # assumindo que tune_* retorna modelos já treinados → extraímos params
#     return {
#         "xgb": best_xgb.get_params(),
#         "rf": best_rf.get_params(),
#         "lr": best_lr.get_params()
#     }


# # ==============================
# # MAIN TEMPORAL
# # ==============================
# def run_baseline_models(df_train, df_val, df_test):

#     print("\n===== BASELINES TEMPORAIS =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     # ==============================
#     # TUNING GLOBAL (IGUAL GNN)
#     # ==============================
#     df_train_init = df_all[df_all["time_step"] < 32]
#     df_val_init = df_all[df_all["time_step"] == 32]

#     best_params = tune_global(df_train_init, df_val_init)

#     # ==============================
#     # STORAGE
#     # ==============================
#     all_probs = {
#         "XGBoost": [],
#         "RandomForest": [],
#         "LogisticRegression": []
#     }

#     all_preds = {
#         "XGBoost": [],
#         "RandomForest": [],
#         "LogisticRegression": []
#     }

#     all_true = []

#     # ==============================
#     # LOOP TEMPORAL
#     # ==============================
#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         df_train_t = df_past[df_past["time_step"] < t]
#         df_val_t = df_past[df_past["time_step"] == t]

#         if len(df_train_t) == 0 or len(df_val_t) == 0:
#             continue

#         X_train = df_train_t[feature_cols]
#         y_train = df_train_t["class"]

#         X_val = df_val_t[feature_cols]
#         y_val = df_val_t["class"]

#         X_train_full = pd.concat([X_train, X_val])
#         y_train_full = pd.concat([y_train, y_val])

#         # ==============================
#         # MODELOS COM PARAMS FIXOS
#         # ==============================
#         model_xgb = XGBClassifier(**best_params["xgb"])
#         model_rf = RandomForestClassifier(**best_params["rf"])
#         model_lr = LogisticRegression(**best_params["lr"])

#         model_xgb.fit(X_train_full, y_train_full)
#         model_rf.fit(X_train_full, y_train_full)
#         model_lr.fit(X_train_full, y_train_full)

#         models = {
#             "XGBoost": model_xgb,
#             "RandomForest": model_rf,
#             "LogisticRegression": model_lr
#         }

#         X_target = df_target[feature_cols]
#         y_target = df_target["class"].values

#         # ==============================
#         # POR MODELO
#         # ==============================
#         for name, model in models.items():

#             probs_val = model.predict_proba(X_val)[:, 1]

#             if len(np.unique(y_val)) >= 2:
#                 best_t = find_best_threshold(y_val, probs_val)
#             else:
#                 best_t = 0.5

#             probs_test = model.predict_proba(X_target)[:, 1]
#             preds_test = (probs_test >= best_t).astype(int)

#             all_probs[name].extend(probs_test)
#             all_preds[name].extend(preds_test)

#         all_true.extend(y_target)

#     y_true = np.array(all_true)

#     # ==============================
#     # RESULTADOS
#     # ==============================
#     results = {}

#     for name in all_probs.keys():

#         y_prob = np.array(all_probs[name])
#         y_pred = np.array(all_preds[name])

#         results[name] = {
#             "PR_AUC": average_precision_score(y_true, y_prob),
#             "F1": f1_score(y_true, y_pred, zero_division=0),
#             "Precision": precision_score(y_true, y_pred, zero_division=0),
#             "Recall": recall_score(y_true, y_pred, zero_division=0),
#         }

#         plot_confusion_matrix(y_true, y_pred, name)
#         plot_pr_curve(y_true, y_prob, name)

#     results_df = pd.DataFrame(results).T

#     print("\n===== RESULTADOS FINAIS =====")
#     print(results_df)

#     results_df.to_csv("results/baseline_temporal_results.csv")

#     return results_df



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

# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# os.makedirs("results/figures", exist_ok=True)


# # ==============================
# # THRESHOLD DINÂMICO
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
# # MAIN TEMPORAL
# # ==============================
# def run_baseline_models_temporal(df_train, df_val, df_test):

#     print("\n===== BASELINES TEMPORAIS =====")

#     df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

#     feature_cols = [
#         c for c in df_all.columns
#         if c not in ["txId", "class", "time_step"]
#     ]

#     df_all["time_step"] = df_all["time_step"].astype(int)
#     max_time = int(df_all["time_step"].max())

#     all_probs = {
#         "XGBoost": [],
#         "RandomForest": [],
#         "LogisticRegression": []
#     }

#     all_preds = {
#         "XGBoost": [],
#         "RandomForest": [],
#         "LogisticRegression": []
#     }

#     all_true = []

#     # ==============================
#     # LOOP TEMPORAL
#     # ==============================
#     for t in range(30, max_time):

#         df_past = df_all[df_all["time_step"] <= t]
#         df_target = df_all[df_all["time_step"] == t + 1]

#         if len(df_target) == 0:
#             continue

#         df_train_t = df_past[df_past["time_step"] < t]
#         df_val_t = df_past[df_past["time_step"] == t]

#         if len(df_train_t) == 0 or len(df_val_t) == 0:
#             continue

#         X_train = df_train_t[feature_cols]
#         y_train = df_train_t["class"]

#         X_val = df_val_t[feature_cols]
#         y_val = df_val_t["class"]

#         X_train_full = pd.concat([X_train, X_val])
#         y_train_full = pd.concat([y_train, y_val])

#         # ==============================
#         # TREINO MODELOS
#         # ==============================
#         model_xgb = tune_xgboost(
#             X_train, y_train, X_val, y_val,
#             X_train_full, y_train_full
#         )

#         model_rf = tune_random_forest(
#             X_train, y_train, X_val, y_val,
#             X_train_full, y_train_full
#         )

#         model_lr = tune_logistic_regression(
#             X_train, y_train, X_val, y_val,
#             X_train_full, y_train_full
#         )

#         models = {
#             "XGBoost": model_xgb,
#             "RandomForest": model_rf,
#             "LogisticRegression": model_lr
#         }

#         X_target = df_target[feature_cols]
#         y_target = df_target["class"].values

#         # ==============================
#         # PROCESSA CADA MODELO
#         # ==============================
#         for name, model in models.items():

#             probs_val = model.predict_proba(X_val)[:, 1]

#             if len(np.unique(y_val)) >= 2:
#                 best_t = find_best_threshold(y_val, probs_val)
#             else:
#                 best_t = 0.5

#             probs_test = model.predict_proba(X_target)[:, 1]
#             preds_test = (probs_test >= best_t).astype(int)

#             all_probs[name].extend(probs_test)
#             all_preds[name].extend(preds_test)

#         all_true.extend(y_target)

#     y_true = np.array(all_true)

#     # ==============================
#     # RESULTADOS FINAIS
#     # ==============================
#     results = {}

#     for name in all_probs.keys():

#         y_prob = np.array(all_probs[name])
#         y_pred = np.array(all_preds[name])

#         results[name] = {
#             "PR_AUC": average_precision_score(y_true, y_prob),
#             "F1": f1_score(y_true, y_pred, zero_division=0),
#             "Precision": precision_score(y_true, y_pred, zero_division=0),
#             "Recall": recall_score(y_true, y_pred, zero_division=0),
#         }

#         plot_confusion_matrix(y_true, y_pred, name)
#         plot_pr_curve(y_true, y_prob, name)

#     results_df = pd.DataFrame(results).T

#     print("\n===== RESULTADOS TEMPORAIS =====")
#     print(results_df)

#     results_df.to_csv("results/baseline_temporal_results.csv")

#     return results_df






##V1
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

# os.makedirs("results/figures", exist_ok=True)


# # ==============================
# # AVALIAÇÃO FINAL (THRESHOLD FIXO)
# # ==============================
# def evaluate(y_true, y_probs, model_name):
#     threshold = 0.5
#     preds = (y_probs >= threshold).astype(int)

#     plot_confusion_matrix(y_true, preds, model_name)
#     plot_pr_curve(y_true, y_probs, model_name)

#     return {
#         "PR_AUC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds, zero_division=0),
#         "Precision": precision_score(y_true, preds, zero_division=0),
#         "Recall": recall_score(y_true, preds, zero_division=0),
#         "Threshold": threshold
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

#     # treino final
#     X_train_full = pd.concat([X_train, X_val])
#     y_train_full = pd.concat([y_train, y_val])

#     # modelos
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

#     # teste
#     probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
#     probs_rf = model_rf.predict_proba(X_test)[:, 1]
#     probs_lr = model_lr.predict_proba(X_test)[:, 1]

#     # avaliação
#     results = {
#         "XGBoost": evaluate(y_test, probs_xgb, "XGBoost"),
#         "RandomForest": evaluate(y_test, probs_rf, "RandomForest"),
#         "LogisticRegression": evaluate(y_test, probs_lr, "LogisticRegression"),
#     }

#     results_df = pd.DataFrame(results).T

#     print("\n===== RESULTADOS FINAIS =====")
#     print(results_df)

#     results_df.to_csv("results/baseline_results.csv")

#     return results_df


## V2
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
# from src.evaluation.plots_summary import plot_metric_comparison
# from src.evaluation.plots import (
#     plot_confusion_matrix,
#     plot_pr_curve
# )

# os.makedirs("results/figures", exist_ok=True)


# # ==============================
# # THRESHOLD OTIMIZADO
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
# # AVALIAÇÃO GENÉRICA (SEM PLOT)
# # ==============================
# def evaluate_simple(y_true, y_probs, threshold):
#     preds = (y_probs >= threshold).astype(int)

#     return {
#         "PR_AUC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds, zero_division=0),
#         "Precision": precision_score(y_true, preds, zero_division=0),
#         "Recall": recall_score(y_true, preds, zero_division=0),
#     }


# # ==============================
# # AVALIAÇÃO FINAL (COM PLOT)
# # ==============================
# def evaluate_test(y_true, y_probs, threshold, model_name):
#     preds = (y_probs >= threshold).astype(int)

#     plot_confusion_matrix(y_true, preds, model_name)
#     plot_pr_curve(y_true, y_probs, model_name)

#     return {
#         "PR_AUC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds, zero_division=0),
#         "Precision": precision_score(y_true, preds, zero_division=0),
#         "Recall": recall_score(y_true, preds, zero_division=0),
#         "Threshold": threshold
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

#     # treino final
#     X_train_full = pd.concat([X_train, X_val])
#     y_train_full = pd.concat([y_train, y_val])

#     # ==============================
#     # MODELOS
#     # ==============================
#     model_xgb = tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
#     model_rf = tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full)
#     model_lr = tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full)

#     # ==============================
#     # PROBABILIDADES
#     # ==============================
#     def get_probs(model):
#         return (
#             model.predict_proba(X_train)[:, 1],
#             model.predict_proba(X_val)[:, 1],
#             model.predict_proba(X_test)[:, 1]
#         )

#     probs_train_xgb, probs_val_xgb, probs_test_xgb = get_probs(model_xgb)
#     probs_train_rf, probs_val_rf, probs_test_rf = get_probs(model_rf)
#     probs_train_lr, probs_val_lr, probs_test_lr = get_probs(model_lr)

#     # ==============================
#     # THRESHOLD (VAL)
#     # ==============================
#     t_xgb = find_best_threshold(y_val, probs_val_xgb)
#     t_rf = find_best_threshold(y_val, probs_val_rf)
#     t_lr = find_best_threshold(y_val, probs_val_lr)

#     # ==============================
#     # AVALIAÇÃO
#     # ==============================
#     results = {}

#     for name, y_p_train, y_p_val, y_p_test, t in [
#         ("XGBoost", probs_train_xgb, probs_val_xgb, probs_test_xgb, t_xgb),
#         ("RandomForest", probs_train_rf, probs_val_rf, probs_test_rf, t_rf),
#         ("LogisticRegression", probs_train_lr, probs_val_lr, probs_test_lr, t_lr),
#     ]:

#         # results[f"{name}_Train"] = evaluate_simple(y_train, y_p_train, t)
#         results[f"{name}_Train"] = {
#             "PR_AUC": average_precision_score(y_train, y_p_train),
#             "F1": np.nan,
#             "Precision": np.nan,
#             "Recall": np.nan,
#         }
#         results[f"{name}_Val"] = evaluate_simple(y_val, y_p_val, t)
#         results[f"{name}_Test"] = evaluate_test(y_test, y_p_test, t, name)

#     # ==============================
#     # OUTPUT
#     # ==============================
#     results_df = pd.DataFrame(results).T

#     print("\n===== RESULTADOS COMPLETOS =====")
#     print(results_df)

#     results_df.to_csv("results/baseline_results_detailed.csv")

#     return results_df

#    # ==============================
#     # GRÁFICOS DE ARTIGO (NOVO)
#     # ==============================
#     plot_metric_comparison(results_df, "Recall", "recall_comparison")
#     plot_metric_comparison(results_df, "Precision", "precision_comparison")
#     plot_metric_comparison(results_df, "F1", "f1_comparison")
#     plot_metric_comparison(results_df, "PR_AUC", "pr_auc_comparison")






## V3
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


