import pandas as pd


def get_elliptic_splits(
    features_path: str,
    classes_path: str,
    train_max_ts: int = 30,
    val_max_ts: int = 34,
    include_unknown: bool = False
):

    # ==============================
    # 1. CARREGAMENTO
    # ==============================
    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)

    # Nomear colunas corretamente
    feature_cols = [f"f{i}" for i in range(features.shape[1] - 2)]
    features.columns = ["txId", "time_step"] + feature_cols

    df = features.merge(classes, on="txId", how="left")

    # ==============================
    # 2. TRATAMENTO DAS CLASSES
    # ==============================
    if not include_unknown:
        df = df[df["class"] != "unknown"]

    df["class"] = df["class"].astype(int)

    # Binário
    df["class"] = df["class"].map({1: 1, 2: 0})

    # ==============================
    # 3. SPLIT TEMPORAL
    # ==============================
    df_train = df[df["time_step"] <= train_max_ts].copy()
    df_val = df[
        (df["time_step"] > train_max_ts) &
        (df["time_step"] <= val_max_ts)
    ].copy()
    df_test = df[df["time_step"] > val_max_ts].copy()

    # ==============================
    # 4. LOG
    # ==============================
    print("\n===== SPLIT TEMPORAL =====")
    print(f"Train: 1 → {train_max_ts} | {len(df_train)}")
    print(f"Val: {train_max_ts+1} → {val_max_ts} | {len(df_val)}")
    print(f"Test: {val_max_ts+1}+ | {len(df_test)}")

    return df_train, df_val, df_test
