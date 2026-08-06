import itertools

import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare


def _load_wide_table(csv_paths, metric, min_timestep):
    series = {}

    for model_name, path in csv_paths.items():
        df = pd.read_csv(path)
        df = df[df["time_step"] >= min_timestep]
        series[model_name] = df.set_index("time_step")[metric]

    wide = pd.concat(series.values(), axis=1, join="inner", keys=series.keys())
    return wide.dropna()


def friedman_test(csv_paths, metric="PR_AUC", min_timestep=35, alpha=0.05):
    wide = _load_wide_table(csv_paths, metric, min_timestep)

    models = list(wide.columns)
    n_blocks = len(wide)

    if n_blocks < 2 or len(models) < 3:
        return {
            "models": models,
            "n_blocks": n_blocks,
            "statistic": None,
            "p_value": None,
            "significativo": False,
        }

    statistic, p_value = friedmanchisquare(*[wide[m] for m in models])

    return {
        "models": models,
        "n_blocks": n_blocks,
        "statistic": statistic,
        "p_value": p_value,
        "significativo": p_value < alpha,
    }


def compare_models(csv_paths, metric="PR_AUC", min_timestep=35, alpha=0.05):
    series = {}

    for model_name, path in csv_paths.items():
        df = pd.read_csv(path)
        df = df[df["time_step"] >= min_timestep]
        series[model_name] = df.set_index("time_step")[metric]

    pairs = list(itertools.combinations(series.keys(), 2))
    n_comparisons = len(pairs) if pairs else 1

    rows = []

    for model_a, model_b in pairs:
        paired = pd.concat(
            [series[model_a], series[model_b]],
            axis=1,
            join="inner",
            keys=[model_a, model_b]
        ).dropna()

        n_pairs = len(paired)

        if n_pairs < 2 or (paired[model_a] == paired[model_b]).all():
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_pares": n_pairs,
                "statistic": None,
                "p_value": None,
                "p_value_bonferroni": None,
                "significativo": False,
            })
            continue

        statistic, p_value = wilcoxon(paired[model_a], paired[model_b])
        p_value_bonferroni = min(p_value * n_comparisons, 1.0)

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "n_pares": n_pairs,
            "statistic": statistic,
            "p_value": p_value,
            "p_value_bonferroni": p_value_bonferroni,
            "significativo": p_value_bonferroni < alpha,
        })

    return pd.DataFrame(rows)
