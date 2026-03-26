''''
python experiments/run_baseline.py

python -m experiments.run_baseline
'''
from src.data.split import get_elliptic_splits
from src.models.baseline import run_baseline_models
from src.utils.config import FEATURES_FILE, CLASSES_FILE


def main():
    df_train, df_val, df_test = get_elliptic_splits(
        FEATURES_FILE,
        CLASSES_FILE
    )

    run_baseline_models(df_train, df_val, df_test)


if __name__ == "__main__":
    main()
