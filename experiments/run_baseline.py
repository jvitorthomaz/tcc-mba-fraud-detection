"""
python -m experiments.run_baseline
"""
"""
Pipeline principal para execução dos modelos baseline.

Inclui:
- Split temporal correto
- Feature engineering sem leakage
- Treinamento de modelos baseline
- Otimização com Optuna
"""

import pandas as pd

from src.data.split import get_elliptic_splits
from src.features.graph_features import add_graph_features
from src.models.baseline import run_baseline_models

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
    run_baseline_models(df_train, df_val, df_test)


if __name__ == "__main__":
    main()

# from src.data.split import get_elliptic_splits
# from src.models.baseline import run_baseline_models
# from src.features.graph_features import add_graph_features

# import pandas as pd
# from src.utils.config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE


# def main():
#     # ==============================
#     # LOAD SPLIT ORIGINAL (SEM FEATURES)
#     # ==============================
#     df_train, df_val, df_test = get_elliptic_splits(
#         FEATURES_FILE,
#         CLASSES_FILE
#     )

#     edges = pd.read_csv(EDGES_FILE)

#     print("\n===== SPLIT TEMPORAL =====")
#     print(f"Train: 1 → 30 | {len(df_train)}")
#     print(f"Val: 31 → 34 | {len(df_val)}")
#     print(f"Test: 35+ | {len(df_test)}")

#     # ==============================
#     # FEATURE ENGINEERING - TRAIN
#     # ==============================
#     print("\n===== FEATURE ENGINEERING - TRAIN =====")
#     df_train_feat = add_graph_features(df_train, edges)

#     # ==============================
#     # FEATURE ENGINEERING - VALIDATION
#     # ==============================
#     print("\n===== FEATURE ENGINEERING - VALIDATION =====")

#     df_val_full = pd.concat(
#         [df_train, df_val],
#         ignore_index=True
#     )

#     df_val_full = add_graph_features(df_val_full, edges)

#     # pega só parte da validação
#     df_val_feat = df_val_full[
#         df_val_full["txId"].isin(df_val["txId"])
#     ].copy()

#     # ==============================
#     # FEATURE ENGINEERING - TEST
#     # ==============================
#     print("\n===== FEATURE ENGINEERING - TEST =====")

#     df_test_full = pd.concat(
#         [df_train, df_val, df_test],
#         ignore_index=True
#     )

#     df_test_full = add_graph_features(df_test_full, edges)

#     # pega só parte do teste
#     df_test_feat = df_test_full[
#         df_test_full["txId"].isin(df_test["txId"])
#     ].copy()

#     # ==============================
#     # TREINAR MODELOS
#     # ==============================
#     run_baseline_models(
#         df_train_feat,
#         df_val_feat,
#         df_test_feat
#     )


# if __name__ == "__main__":
#     main()
