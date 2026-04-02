import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.models.utils import find_best_threshold


def run_baseline_models(df_train, df_val, df_test):
    target = "class"

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    X_val = df_val.drop(columns=[target])
    y_val = df_val[target]

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # Garantir nomes string
    X_train.columns = X_train.columns.astype(str)
    X_val.columns = X_val.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # =========================
    # Class weights
    # =========================
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    results = []

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight=class_weight_dict),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight=class_weight_dict),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=weights[1] / weights[0],
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    for name, model in models.items():
        print(f"Treinando {name}...")
        model.fit(X_train, y_train)

        # Validação → threshold tuning
        y_val_prob = model.predict_proba(X_val)[:, 1]
        best_thresh = find_best_threshold(y_val, y_val_prob)

        # Teste
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= best_thresh).astype(int)

        pr_auc = average_precision_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "model": name,
            "PR_AUC": pr_auc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall
        })

    df_results = pd.DataFrame(results)
    print("\n===== RESULTADOS =====")
    print(df_results)

    df_results.to_csv("results/new_tables/baseline_results.csv", index=False)

    print("\nResultados salvos em: results/new_tables/baseline_results.csv")
