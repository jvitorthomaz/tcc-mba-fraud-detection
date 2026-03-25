import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve
)


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def prepare_data(df):
    X = df.drop(columns=["txId", "class", "time_step"])
    y = df["class"]
    return X, y


def run_baseline_models(df_train, df_val, df_test):
    print("\n===== BASELINE MODELS =====")

    # ==============================
    # 1. PREPARAÇÃO
    # ==============================
    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)
    X_test, y_test = prepare_data(df_test)

    # Normalização (importante para LogReg)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = []

    # ==============================
    # 2. LOGISTIC REGRESSION
    # ==============================
    print("\nTreinando Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = "LogisticRegression"
    results.append(metrics)

    plot_confusion_matrix(y_test, y_pred, "LogisticRegression")
    plot_roc_curve(y_test, y_prob, "LogisticRegression")
    plot_pr_curve(y_test, y_prob, "LogisticRegression")

    # ==============================
    # 3. RANDOM FOREST
    # ==============================
    print("Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = "RandomForest"
    results.append(metrics)

    plot_confusion_matrix(y_test, y_pred, "RandomForest")
    plot_roc_curve(y_test, y_prob, "RandomForest")
    plot_pr_curve(y_test, y_prob, "RandomForest")

    # ==============================
    # 4. XGBOOST
    # ==============================
    print("Treinando XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["model"] = "XGBoost"
    results.append(metrics)

    plot_confusion_matrix(y_test, y_pred, "XGBoost")
    plot_roc_curve(y_test, y_prob, "XGBoost")
    plot_pr_curve(y_test, y_prob, "XGBoost")

    # ==============================
    # 5. RESULTADOS
    # ==============================
    results_df = pd.DataFrame(results)

    print("\n===== RESULTADOS =====")
    print(results_df)

    results_df.to_csv("results/tables/baseline_results.csv", index=False)

    print("\nResultados salvos em: results/tables/baseline_results.csv")

    return results_df
