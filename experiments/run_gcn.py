''''

python -m experiments.run_gcn

'''

import pandas as pd

from src.data.split import get_elliptic_splits
from src.models.gcn import run_gcn
from src.utils.config import (
    FEATURES_FILE,
    CLASSES_FILE,
    EDGES_FILE,
    TRAIN_MAX_TS,
    VAL_MAX_TS
)


def main():

    # ==============================
    # SPLIT TEMPORAL
    # ==============================

    df_train, df_val, df_test = get_elliptic_splits(
        features_path=FEATURES_FILE,
        classes_path=CLASSES_FILE,
        train_max_ts=TRAIN_MAX_TS,
        val_max_ts=VAL_MAX_TS,
        include_unknown=False
    )

    # ==============================
    # EDGES
    # ==============================

    edges = pd.read_csv(EDGES_FILE)

    # ==============================
    # GCN
    # ==============================

    run_gcn(df_train, df_val, df_test, edges)


if __name__ == "__main__":
    main()
