import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

def tune_random_forest(X_train, y_train, X_val, y_val, X_train_full, y_train_full):

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

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # study.optimize(objective, n_trials=30)

    print("\n===== MELHORES PARÂMETROS RANDOM FOREST =====")
    print(study.best_params)
    print(study.best_params, "| PR-AUC:", round(study.best_value, 4))

    best_model = RandomForestClassifier(
        **study.best_params,
        n_jobs=-1,
        random_state=42
    )

    best_model.fit(X_train_full, y_train_full)

    return best_model
