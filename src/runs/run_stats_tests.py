"""
python -m src.runs.run_stats_tests
"""

import glob
import os

import pandas as pd

from src.evaluation.stats_tests import compare_models, friedman_test


def main():
    csv_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob.glob("results/per_timestep/*.csv")
    }

    if len(csv_paths) < 2:
        print("É preciso pelo menos 2 modelos com CSV em results/per_timestep/ para comparar.")
        return

    print("Modelos encontrados:", list(csv_paths.keys()))

    os.makedirs("results", exist_ok=True)

    friedman = friedman_test(csv_paths)

    print("\n===== TESTE GLOBAL (Friedman, PR_AUC, time_step>=35) =====")
    print(friedman)

    pd.DataFrame([friedman]).to_csv("results/friedman_test.csv", index=False)

    if len(friedman["models"]) < 3:
        print("\nMenos de 3 modelos com dados alinhados — Friedman não é aplicável, pulando o teste global.")
    elif not friedman["significativo"]:
        print("\nFriedman não encontrou diferença global significativa — comparações par a par (Wilcoxon) não serão feitas.")
        return

    comparison = compare_models(csv_paths)

    print("\n===== COMPARAÇÃO ESTATÍSTICA (Wilcoxon, PR_AUC, time_step>=35) =====")
    print(comparison.to_string(index=False))

    comparison.to_csv("results/statistical_comparison.csv", index=False)


if __name__ == "__main__":
    main()
