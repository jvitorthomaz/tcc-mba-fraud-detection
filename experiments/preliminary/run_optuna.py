"""
Execução dedicada dos modelos com Optuna.

Inclui:
- Split temporal
- Feature engineering sem leakage
- Treinamento com Optuna (XGB, RF, LR)
- Avaliação final
"""

import pandas as pd

from src.data.split import get_elliptic_splits
from src.features.graph_features import add_graph_features

from src.models.optuna_xgboost import tune_xgboost
from src.models.optuna_random_forest import tune_random_forest
from src.models.optuna_logistic_regression import tune_logistic_regression

from src.models.baseline import evaluate

from src.utils.config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE


def main():

    # ==============================
    # SPLIT TEMPORAL
    # ==============================
    df_train, df_val, df_test = get_elliptic_splits(
        FEATURES_FILE,
        CLASSES_FILE
    )

    edges = pd.read_csv(EDGES_FILE)

    # ==============================
    # FEATURE ENGINEERING SEM LEAKAGE
    # ==============================

    print("\n===== FEATURE ENGINEERING - TRAIN =====")
    df_train = add_graph_features(df_train, edges)

    print("\n===== FEATURE ENGINEERING - VALIDATION =====")
    df_val_full = pd.concat([df_train, df_val], ignore_index=True)
    df_val_full = add_graph_features(df_val_full, edges)
    df_val = df_val_full[df_val_full["time_step"] > 30].reset_index(drop=True)

    print("\n===== FEATURE ENGINEERING - TEST =====")
    df_test_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_test_full = add_graph_features(df_test_full, edges)
    df_test = df_test_full[df_test_full["time_step"] > 34].reset_index(drop=True)

    # ==============================
    # FEATURES
    # ==============================

    feature_cols = [c for c in df_train.columns if c not in ["txId", "class", "time_step"]]

    X_train = df_train[feature_cols]
    y_train = df_train["class"]

    X_val = df_val[feature_cols]
    y_val = df_val["class"]

    X_test = df_test[feature_cols]
    y_test = df_test["class"]

    # ==============================
    # OPTUNA MODELS
    # ==============================

    print("\n===== OTIMIZANDO XGBOOST =====")
    model_xgb = tune_xgboost(X_train, y_train, X_val, y_val)

    print("\n===== OTIMIZANDO RANDOM FOREST =====")
    model_rf = tune_random_forest(X_train, y_train, X_val, y_val)

    print("\n===== OTIMIZANDO LOGISTIC REGRESSION =====")
    model_lr = tune_logistic_regression(X_train, y_train, X_val, y_val)

    # ==============================
    # PREDIÇÕES
    # ==============================

    probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
    probs_rf = model_rf.predict_proba(X_test)[:, 1]
    probs_lr = model_lr.predict_proba(X_test)[:, 1]

    probs_ensemble = (probs_xgb + probs_rf + probs_lr) / 3

    # ==============================
    # RESULTADOS
    # ==============================

    results = {
        "XGBoost": evaluate(y_test, probs_xgb),
        "RandomForest": evaluate(y_test, probs_rf),
        "LogisticRegression": evaluate(y_test, probs_lr),
        "Ensemble": evaluate(y_test, probs_ensemble),
    }

    print("\n===== RESULTADOS FINAIS (OPTUNA) =====")
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()
