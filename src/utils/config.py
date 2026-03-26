# src/utils/config.py

DATA_PATH = "data/raw/"

FEATURES_FILE = DATA_PATH + "elliptic_txs_features.csv"
CLASSES_FILE = DATA_PATH + "elliptic_txs_classes.csv"
EDGES_FILE = DATA_PATH + "elliptic_txs_edgelist.csv"

# Split temporal
TRAIN_MAX_TS = 30
VAL_MAX_TS = 34

# Reprodutibilidade
RANDOM_SEED = 42

# GCN
GCN_EPOCHS = 50
GCN_LR = 0.01
GCN_HIDDEN_1 = 64
GCN_HIDDEN_2 = 32
