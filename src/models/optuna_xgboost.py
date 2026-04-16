"""
Otimização de hiperparâmetros do XGBoost com Optuna.

Inclui:
- Early stopping correto
- Otimização baseada em PR-AUC
- Regularização forte para evitar overfitting
"""

"""
Otimização de hiperparâmetros do XGBoost com Optuna
"""

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score


def compute_scale_pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()

    if pos == 0:
        return 1.0

    return neg / pos


def tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full):

    base_scale = compute_scale_pos_weight(y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),

            # IMPORTANTE
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                base_scale * 0.5,
                base_scale * 2
            ),

            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50
        }

        model = XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        probs = model.predict_proba(X_val)[:, 1]

        return average_precision_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)
    # study.optimize(objective, n_trials=30)

    print("\n===== MELHORES PARÂMETROS XGBOOST =====")
    print(study.best_params)
    print(study.best_params, "| PR-AUC:", round(study.best_value, 4))

    best_model = XGBClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    best_model.fit(X_train_full, y_train_full)

    return best_model


# def tune_xgboost(X_train, y_train, X_val, y_val, X_train_full, y_train_full):

#     def objective(trial):
#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 300, 900),
#             "max_depth": trial.suggest_int("max_depth", 4, 10),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
#             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#             "gamma": trial.suggest_float("gamma", 0, 5),
#             "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),

#             # Config fixo
#             "random_state": 42,
#             "n_jobs": -1,
#             "eval_metric": "logloss",
#             "early_stopping_rounds": 50  # agora no construtor (correto)
#         }

#         model = XGBClassifier(**params)

#         model.fit(
#             X_train,
#             y_train,
#             eval_set=[(X_val, y_val)],
#             verbose=False
#         )

#         probs = model.predict_proba(X_val)[:, 1]

#         return average_precision_score(y_val, probs)

#     # ==============================
#     # OPTUNA
#     # ==============================

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=50)

#     print("\n===== MELHORES PARÂMETROS XGBOOST =====")
#     print(study.best_params)

#     # ==============================
#     # MODELO FINAL (TREINO COMPLETO)
#     # ==============================

#     best_model = XGBClassifier(
#         **study.best_params,
#         random_state=42,
#         n_jobs=-1,
#         eval_metric="logloss"
#     )

#     best_model.fit(X_train_full, y_train_full)

#     return best_model

# import optuna
# import xgboost as xgb

# from sklearn.metrics import average_precision_score


# def tune_xgboost(X_train, y_train, X_val, y_val):

#     def objective(trial):

#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 300, 900),
#             "max_depth": trial.suggest_int("max_depth", 4, 10),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
#             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#             "gamma": trial.suggest_float("gamma", 0, 5),
#             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),

#             # Fixos importantes
#             "objective": "binary:logistic",
#             "eval_metric": "logloss",
#             "tree_method": "hist",
#             "random_state": 42,
#             "n_jobs": -1
#         }

#         model = xgb.XGBClassifier(**params)

#         model.fit(
#             X_train,
#             y_train,
#             eval_set=[(X_val, y_val)],
#             verbose=False,
#             early_stopping_rounds=50
#         )

#         probs = model.predict_proba(X_val)[:, 1]

#         score = average_precision_score(y_val, probs)

#         return score

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=50)

#     print("\n===== MELHORES PARÂMETROS XGBOOST =====")
#     print(study.best_params)

#     best_model = xgb.XGBClassifier(
#         **study.best_params,
#         objective="binary:logistic",
#         eval_metric="logloss",
#         tree_method="hist",
#         random_state=42,
#         n_jobs=-1
#     )

#     # Treina modelo final com early stopping
#     best_model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=False,
#         early_stopping_rounds=50
#     )

#     return best_model

