''''

python -m experiments.run_graphsage

'''
from src.data.split import get_elliptic_splits
from src.models.graphsage import run_graphsage
from src.features.graph_features import add_graph_features

import pandas as pd
from src.utils.config import FEATURES_FILE, CLASSES_FILE, EDGES_FILE


def main():
    df_train, df_val, df_test = get_elliptic_splits(
        FEATURES_FILE,
        CLASSES_FILE
    )

    edges = pd.read_csv(EDGES_FILE)

    # FEATURE ENGINEERING
    df_all = pd.concat([df_train, df_val, df_test])
    df_all = add_graph_features(df_all, edges)

    df_train = df_all[df_all["time_step"] <= 30]
    df_val = df_all[(df_all["time_step"] > 30) & (df_all["time_step"] <= 34)]
    df_test = df_all[df_all["time_step"] > 34]

    run_graphsage(df_train, df_val, df_test, edges)


if __name__ == "__main__":
    main()
