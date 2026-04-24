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
