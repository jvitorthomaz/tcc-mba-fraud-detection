import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def tune_logistic_regression(df_train, df_val):

    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)
    feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10, log=True)

        class_weight = trial.suggest_categorical(
            "class_weight",
            [None, "balanced"]
        )

        scores = []

        for t in [32, 33, 34]:
            df_t_train = df_all[df_all["time_step"] < t]
            df_t_val = df_all[df_all["time_step"] == t]

            if len(df_t_train) == 0 or len(df_t_val) == 0:
                continue

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    n_jobs=-1
                ))
            ])

            model.fit(df_t_train[feature_cols], df_t_train["class"])

            probs = model.predict_proba(df_t_val[feature_cols])[:, 1]
            scores.append(average_precision_score(df_t_val["class"], probs))

        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction="maximize")
    #study.optimize(objective, n_trials=50)
    study.optimize(objective, n_trials=2)


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

    best_model.fit(df_all[feature_cols], df_all["class"])

    return best_model
