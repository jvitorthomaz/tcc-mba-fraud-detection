import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from sklearn.metrics import average_precision_score

def tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full):

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10, log=True)

        class_weight = trial.suggest_categorical(
            "class_weight",
            [None, "balanced"]
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=2000,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\n===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====")
    print(study.best_params)
    print(study.best_params, "| PR-AUC:", round(study.best_value, 4))

    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            **study.best_params,
            max_iter=2000,
            n_jobs=-1
        ))
    ])

    best_model.fit(X_train_full, y_train_full)

    return best_model


# def tune_logistic_regression(X_train, y_train, X_val, y_val, X_train_full, y_train_full):

#     def objective(trial):
#         C = trial.suggest_float("C", 1e-3, 10, log=True)

#         model = Pipeline([
#             ("scaler", StandardScaler()),
#             ("lr", LogisticRegression(
#                 C=C,
#                 max_iter=2000,
#                 n_jobs=-1
#             ))
#         ])

#         model.fit(X_train, y_train)

#         probs = model.predict_proba(X_val)[:, 1]
#         return average_precision_score(y_val, probs)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=30)

#     print("\n===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====")
#     print(study.best_params)

#     # TREINO FINAL
#     best_model = Pipeline([
#         ("scaler", StandardScaler()),
#         ("lr", LogisticRegression(
#             **study.best_params,
#             max_iter=2000,
#             n_jobs=-1
#         ))
#     ])

#     best_model.fit(X_train_full, y_train_full)

#     return best_model


# import optuna
# import numpy as np
# import pandas as pd

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import average_precision_score, f1_score
# from sklearn.preprocessing import StandardScaler

# from src.models.utils import find_best_threshold


# def run_optuna_logreg(df_train, df_val, df_test, n_trials=50):

#     target = "class"
#     cols_to_drop = ["class", "txId", "time_step"]

#     X_train = df_train.drop(columns=cols_to_drop)
#     y_train = df_train[target]

#     X_val = df_val.drop(columns=cols_to_drop)
#     y_val = df_val[target]

#     X_test = df_test.drop(columns=cols_to_drop)
#     y_test = df_test[target]

#     # =========================
#     # NORMALIZAÇÃO (OBRIGATÓRIO)
#     # =========================
#     scaler = StandardScaler()

#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     # =========================
#     # OBJETIVO
#     # =========================
#     def objective(trial):

#         params = {
#             "C": trial.suggest_float("C", 1e-3, 10, log=True),
#             "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
#             "solver": "liblinear",
#             "class_weight": "balanced",
#             "max_iter": 1000
#         }

#         model = LogisticRegression(**params)

#         model.fit(X_train, y_train)

#         y_val_prob = model.predict_proba(X_val)[:, 1]

#         return average_precision_score(y_val, y_val_prob)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials)

#     print("\n===== MELHORES PARÂMETROS LOGREG =====")
#     print(study.best_params)

#     # =========================
#     # TREINO FINAL
#     # =========================
#     df_train_full = pd.concat([df_train, df_val])

#     X_train_full = scaler.fit_transform(
#         df_train_full.drop(columns=cols_to_drop)
#     )
#     y_train_full = df_train_full[target]

#     best_model = LogisticRegression(
#         **study.best_params,
#         solver="liblinear",
#         class_weight="balanced",
#         max_iter=1000
#     )

#     best_model.fit(X_train_full, y_train_full)

#     # =========================
#     # TESTE FINAL
#     # =========================
#     X_test_scaled = scaler.transform(X_test)

#     y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]

#     thresh = find_best_threshold(y_test, y_test_prob)
#     y_test_pred = (y_test_prob >= thresh).astype(int)

#     result = {
#         "PR_AUC": average_precision_score(y_test, y_test_prob),
#         "F1": f1_score(y_test, y_test_pred)
#     }

#     print("\n===== RESULTADO FINAL LOGREG =====")
#     print(result)

#     return best_model, study
