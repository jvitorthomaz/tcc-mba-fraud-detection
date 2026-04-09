import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


def tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "n_jobs": -1,
            "random_state": 42
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\n===== MELHORES PARÂMETROS RANDOM FOREST =====")
    print(study.best_params)

    # TREINAMENTO FINAL
    best_model = RandomForestClassifier(
        **study.best_params,
        n_jobs=-1,
        random_state=42
    )

    best_model.fit(X_train_full, y_train_full)

    return best_model



# import optuna
# import numpy as np
# import pandas as pd

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import average_precision_score, f1_score

# from src.models.utils import find_best_threshold


# def run_optuna_rf(df_train, df_val, df_test, n_trials=50):

#     target = "class"
#     cols_to_drop = ["class", "txId", "time_step"]

#     X_train = df_train.drop(columns=cols_to_drop)
#     y_train = df_train[target]

#     X_val = df_val.drop(columns=cols_to_drop)
#     y_val = df_val[target]

#     X_test = df_test.drop(columns=cols_to_drop)
#     y_test = df_test[target]

#     # =========================
#     # OBJETIVO
#     # =========================
#     def objective(trial):

#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 200, 800),
#             "max_depth": trial.suggest_int("max_depth", 5, 30),
#             "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
#             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
#             "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
#             "class_weight": "balanced",
#             "n_jobs": -1,
#             "random_state": 42
#         }

#         model = RandomForestClassifier(**params)

#         model.fit(X_train, y_train)

#         y_val_prob = model.predict_proba(X_val)[:, 1]

#         return average_precision_score(y_val, y_val_prob)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials)

#     print("\n===== MELHORES PARÂMETROS RF =====")
#     print(study.best_params)

#     # =========================
#     # TREINO FINAL
#     # =========================
#     df_train_full = pd.concat([df_train, df_val])

#     X_train_full = df_train_full.drop(columns=cols_to_drop)
#     y_train_full = df_train_full[target]

#     best_model = RandomForestClassifier(
#         **study.best_params,
#         class_weight="balanced",
#         n_jobs=-1,
#         random_state=42
#     )

#     best_model.fit(X_train_full, y_train_full)

#     # =========================
#     # TESTE FINAL
#     # =========================
#     y_test_prob = best_model.predict_proba(X_test)[:, 1]

#     thresh = find_best_threshold(y_test, y_test_prob)
#     y_test_pred = (y_test_prob >= thresh).astype(int)

#     result = {
#         "PR_AUC": average_precision_score(y_test, y_test_prob),
#         "F1": f1_score(y_test, y_test_pred)
#     }

#     print("\n===== RESULTADO FINAL RF =====")
#     print(result)

#     return best_model, study
