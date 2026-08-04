import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


def tune_random_forest(df_train, df_val):

    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)
    feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

    def objective(trial):
        params = {
            # "n_estimators": trial.suggest_int("n_estimators", 300, 1000)
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", 0.5, 0.7, 1.0]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"]
            ),
            "n_jobs": -1,
            "random_state": 42
        }

        scores = []

        for t in [32, 33, 34]:
            df_t_train = df_all[df_all["time_step"] < t]
            df_t_val = df_all[df_all["time_step"] == t]

            if len(df_t_train) == 0 or len(df_t_val) == 0:
                continue

            model = RandomForestClassifier(**params)
            model.fit(df_t_train[feature_cols], df_t_train["class"])

            probs = model.predict_proba(df_t_val[feature_cols])[:, 1]
            scores.append(average_precision_score(df_t_val["class"], probs))

        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction="maximize")
    #study.optimize(objective, n_trials=100)
    study.optimize(objective, n_trials=5)


    print("\n===== MELHORES PARÂMETROS RANDOM FOREST =====")
    print(study.best_params)
    print(study.best_params, "| PR-AUC:", round(study.best_value, 2))

    best_model = RandomForestClassifier(
        **study.best_params,
        n_jobs=-1,
        random_state=42
    )

    best_model.fit(df_all[feature_cols], df_all["class"])

    return best_model
