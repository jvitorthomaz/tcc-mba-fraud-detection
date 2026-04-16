''''
python -m experiments.run_optuna_xgboost
'''

import pandas as pd

from src.data.split import get_elliptic_splits
from src.models.optuna_xgboost import tune_xgboost
from src.features.graph_features import add_graph_features
from src.utils.config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE

def main():
    # ==============================
    # LOAD SPLIT ORIGINAL (SEM FEATURES)
    # ==============================
    df_train, df_val, df_test = get_elliptic_splits(
        FEATURES_FILE,
        CLASSES_FILE
    )

    edges = pd.read_csv(EDGES_FILE)

    print("\n===== SPLIT TEMPORAL =====")
    print(f"Train: 1 → 30 | {len(df_train)}")
    print(f"Val: 31 → 34 | {len(df_val)}")
    print(f"Test: 35+ | {len(df_test)}")

    # ==============================
    # FEATURE ENGINEERING - TRAIN
    # ==============================
    print("\n===== FEATURE ENGINEERING - TRAIN =====")
    df_train_feat = add_graph_features(df_train, edges)

    # ==============================
    # FEATURE ENGINEERING - VALIDATION
    # ==============================
    print("\n===== FEATURE ENGINEERING - VALIDATION =====")

    df_val_full = pd.concat(
        [df_train, df_val],
        ignore_index=True
    )

    df_val_full = add_graph_features(df_val_full, edges)

    # pega só parte da validação
    df_val_feat = df_val_full[
        df_val_full["txId"].isin(df_val["txId"])
    ].copy()

    # ==============================
    # FEATURE ENGINEERING - TEST
    # ==============================
    print("\n===== FEATURE ENGINEERING - TEST =====")

    df_test_full = pd.concat(
        [df_train, df_val, df_test],
        ignore_index=True
    )

    df_test_full = add_graph_features(df_test_full, edges)

    # pega só parte do teste
    df_test_feat = df_test_full[
        df_test_full["txId"].isin(df_test["txId"])
    ].copy()

    # ==============================
    # TRAIN MODEL
    # ==============================
    tune_xgboost(
        df_train_feat,
        df_val_feat,
        df_test_feat
    )
    

if __name__ == "__main__":
    main()
