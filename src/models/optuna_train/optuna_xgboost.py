import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score


def compute_scale_pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()

    if pos == 0:
        return 1.0

    return neg / pos


def tune_xgboost(df_train, df_val):

    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)
    feature_cols = [c for c in df_all.columns if c not in ["txId", "class", "time_step"]]

    base_scale = compute_scale_pos_weight(df_all[df_all["time_step"] < 32]["class"])

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

        scores = []
        best_iterations = []

        for t in [32, 33, 34]:
            df_t_train = df_all[df_all["time_step"] < t]
            df_t_val = df_all[df_all["time_step"] == t]

            if len(df_t_train) == 0 or len(df_t_val) == 0:
                continue

            model = XGBClassifier(**params)

            model.fit(
                df_t_train[feature_cols],
                df_t_train["class"],
                eval_set=[(df_t_val[feature_cols], df_t_val["class"])],
                verbose=False
            )

            best_iterations.append(model.best_iteration)

            probs = model.predict_proba(df_t_val[feature_cols])[:, 1]
            scores.append(average_precision_score(df_t_val["class"], probs))

        trial.set_user_attr("best_iterations", best_iterations)

        return np.mean(scores) if scores else 0

    study = optuna.create_study(direction="maximize")
    #study.optimize(objective, n_trials=150)
    study.optimize(objective, n_trials=2)

    print("\n===== MELHORES PARÂMETROS XGBOOST =====")
    print(study.best_params)
    print(study.best_params, "| PR-AUC:", round(study.best_value, 4))

    best_n_estimators = round(np.mean(study.best_trial.user_attrs["best_iterations"])) + 1
    final_params = {**study.best_params, "n_estimators": best_n_estimators}

    print("n_estimators final (média das 3 janelas via early stopping):", best_n_estimators)

    best_model = XGBClassifier(
        **final_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    best_model.fit(df_all[feature_cols], df_all["class"])

    return best_model
