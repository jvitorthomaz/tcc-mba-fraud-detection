from src.data.split import get_elliptic_splits
from src.models.preliminary.graph_transformer import run_graph_transformer

import pandas as pd
from src.utils.config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE


def main():
    df_train, df_val, df_test = get_elliptic_splits(
        FEATURES_FILE,
        CLASSES_FILE
    )

    edges = pd.read_csv(EDGES_FILE)

    run_graph_transformer(df_train, df_val, df_test, edges)


if __name__ == "__main__":
    main()
