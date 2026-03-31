
## Tabular vs GCN

| Modelo              | PR-AUC   | F1       | Precision | Recall   |
| ------------------- | -------- | -------- | --------- | -------- |
| Logistic Regression | 0.21     | 0.25     | 0.14      | **0.94** |
| Random Forest       | 0.78     | **0.80** | **0.98**  | 0.67     |
| XGBoost             | **0.80** | 0.78     | 0.86      | 0.72     |



| Modelo    | PR-AUC    | F1       | Precision | Recall   |
| --------- | --------- | -------- | --------- | -------- |
| GCN       | 0.246     | 0.32     | 0.20      | **0.72** |
| GraphSAGE | 0.356     | **0.46** | 0.35      | 0.68     |
| GAT       | **0.457** | 0.29     | 0.18      | **0.72** |




## GCN vs GraphSAGE
| Métrica     | GCN      | GraphSAGE |
| ----------- | -------- | --------- |
| Val PR-AUC  | 0.75     | **0.87**  |
| Test PR-AUC | **0.58** | 0.28      |


# Resultados EvolveGCN
| Métrica   | Valor     |
| --------- | --------- |
| PR-AUC    | **0.164** |
| F1        | 0.17      |
| Precision | **0.09**  |
| Recall    | **0.96**  |

