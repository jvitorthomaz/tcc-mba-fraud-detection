import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("results/figures", exist_ok=True)


def plot_metric_comparison(results_df, metric, filename):

    models = ["XGBoost", "RandomForest", "LogisticRegression"]

    train_vals = []
    val_vals = []
    test_vals = []

    for model in models:
        # train_vals.append(results_df.loc[f"{model}_Train", metric])
        val_vals.append(results_df.loc[f"{model}_Val", metric])
        test_vals.append(results_df.loc[f"{model}_Test", metric])

    x = range(len(models))

    plt.figure()

    width = 0.25

    plt.bar([i - width for i in x], train_vals, width=width, label="Train")
    plt.bar(x, val_vals, width=width, label="Validação")
    plt.bar([i + width for i in x], test_vals, width=width, label="Teste")

    plt.xticks(x, models)
    plt.ylabel(metric)
    plt.title(f"Comparação de {metric} entre conjuntos")

    plt.legend()

    plt.savefig(f"results/figures/{filename}.png")
    plt.close()