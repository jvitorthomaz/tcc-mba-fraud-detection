''''
split temporal dos dados
'''
import pandas as pd


def get_elliptic_splits(
    features_path: str,
    classes_path: str,
    train_max_ts: int = 30,
    val_max_ts: int = 34,
    include_unknown: bool = False
):
    """
    Realiza split temporal no Elliptic Dataset.

    Parâmetros:
    ----------
    features_path : str
        Caminho para elliptic_txs_features.csv

    classes_path : str
        Caminho para elliptic_txs_classes.csv

    train_max_ts : int
        Último timestep do treino

    val_max_ts : int
        Último timestep da validação

    include_unknown : bool
        Se True, mantém classe 'unknown'

    Retorno:
    -------
    df_train, df_val, df_test : pd.DataFrame
    """

    # ==============================
    # 1. CARREGAMENTO
    # ==============================
    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)

    features = features.rename(columns={
        0: "txId",
        1: "time_step"
    })

    df = features.merge(classes, on="txId", how="left")

    # ==============================
    # 2. TRATAMENTO DAS CLASSES
    # ==============================
    if not include_unknown:
        df = df[df["class"] != "unknown"]

    df["class"] = df["class"].astype(int)

    # Converter para binário (XGBoost exige 0 e 1)
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
    print(f"Train: 1 → {train_max_ts} | {len(df_train)} amostras")
    print(f"Val: {train_max_ts+1} → {val_max_ts} | {len(df_val)} amostras")
    print(f"Test: {val_max_ts+1} → {df['time_step'].max()} | {len(df_test)} amostras")

    # Distribuição de classes
    print("\nDistribuição de classes (Train):")
    print(df_train["class"].value_counts(normalize=True))

    return df_train, df_val, df_test
