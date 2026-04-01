import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)


def prepare_data(df):
    X = df.drop(columns=["txId", "class"])
    y = df["class"]

    # 🔴 CORREÇÃO PRINCIPAL
    X.columns = X.columns.astype(str)

    return X, y


def evaluate_model(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    return {
        "PR_AUC": average_precision_score(y, y_prob),
        "F1": f1_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
    }


def run_baseline_models(df_train, df_val, df_test):

    print("\n===== BASELINE MODELS =====")

    # ==============================
    # PREPARAÇÃO
    # ==============================

    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)
    X_test, y_test = prepare_data(df_test)

    # Normalização (apenas para modelos lineares)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ==============================
    # LOGISTIC REGRESSION
    # ==============================

    print("\nTreinando Logistic Regression...")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    metrics = evaluate_model(lr, X_test_scaled, y_test)
    metrics["model"] = "LogisticRegression"
    results.append(metrics)

    # ==============================
    # RANDOM FOREST
    # ==============================

    print("Treinando Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train, y_train)

    metrics = evaluate_model(rf, X_test, y_test)
    metrics["model"] = "RandomForest"
    results.append(metrics)

    # ==============================
    # XGBOOST
    # ==============================

    print("Treinando XGBoost...")

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    metrics = evaluate_model(xgb, X_test, y_test)
    metrics["model"] = "XGBoost"
    results.append(metrics)

    # ==============================
    # RESULTADOS
    # ==============================

    results_df = pd.DataFrame(results)

    print("\n===== RESULTADOS =====")
    print(results_df)

    results_df.to_csv("results/new_tables/baseline_results.csv", index=False)

    print("\nResultados salvos em: results/new_tables/baseline_results.csv")
