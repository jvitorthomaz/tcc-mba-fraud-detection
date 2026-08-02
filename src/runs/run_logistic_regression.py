"""
python -m src.runs.run_logistic_regression
"""

import pandas as pd

from src.data.split import get_elliptic_splits
from src.features.graph_features import add_graph_features
from src.models.logistic_regression import run_logistic_regression

from src.utils.config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE


def main():
    # ==============================
    # SPLIT TEMPORAL ORIGINAL
    # ==============================
    df_train_raw, df_val_raw, df_test_raw = get_elliptic_splits(
        FEATURES_FILE,
        CLASSES_FILE
    )

    edges = pd.read_csv(EDGES_FILE)

    # ==============================
    # FEATURE ENGINEERING - TRAIN
    # ==============================
    print("\n===== FEATURE ENGINEERING - TRAIN =====")
    df_train = add_graph_features(df_train_raw, edges)

    # ==============================
    # FEATURE ENGINEERING - VALIDATION
    # ==============================
    print("\n===== FEATURE ENGINEERING - VALIDATION =====")

    df_val_input = pd.concat(
        [df_train_raw, df_val_raw],
        ignore_index=True
    )

    df_val_input = add_graph_features(df_val_input, edges)

    df_val = df_val_input[
        df_val_input["time_step"] > 30
    ].reset_index(drop=True)

    # ==============================
    # FEATURE ENGINEERING - TEST
    # ==============================
    print("\n===== FEATURE ENGINEERING - TEST =====")

    df_test_input = pd.concat(
        [df_train_raw, df_val_raw, df_test_raw],
        ignore_index=True
    )

    df_test_input = add_graph_features(df_test_input, edges)

    df_test = df_test_input[
        df_test_input["time_step"] > 34
    ].reset_index(drop=True)

    # ==============================
    # TREINAMENTO
    # ==============================
    run_logistic_regression(df_train, df_val, df_test)


if __name__ == "__main__":
    main()
