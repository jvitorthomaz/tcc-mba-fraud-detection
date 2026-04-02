# Comparative Table

## Baseline
| Modelo              | PR-AUC | F1-score | Precision | Recall |
| ------------------- | ------ | -------- | --------- | ------ |
| Logistic Regression | 0.212  | 0.252    | 0.145     | 0.944  |
| Random Forest       | 0.783  | 0.800    | 0.986     | 0.673  |
| XGBoost             | 0.799  | 0.784    | 0.861     | 0.719  |
| GCN                 | 0.247  | 0.320    | 0.206     | 0.721  |
| GraphSAGE           | 0.356  | 0.460    | 0.346     | 0.684  |
| GAT                 | 0.457  | 0.292    | 0.183     | 0.721  |
| EvolveGCN           | 0.164  | 0.171    | 0.094     | 0.967  |
| TGN                 | 0.207  | 0.248    | 0.144     | 0.900  |


## Liste de Ajustes

| Etapa              | Impacto | Prioridade |
| ------------------ | ------- | ---------- |
| Graph features     | 3       | Feito      |
| Threshold tuning   | 3       | Feito      |
| Class weights      | 3       | Feito      |
| Early stopping     | 3       | Feito      |
| XGBoost tuning     | 3       | Próxima    |
| GraphSAGE tuning   | 2       | Média      |
| Regularização GNN  | 1       | Média      |
| Features temporais | 2       | Depois     |
| Pipeline temporal  | 3       | Opcional   |




## Rodada 1 de Ajustes

| Modelo              | PR-AUC | F1-score | Precision | Recall |
| ------------------- | ------ | -------- | --------- | ------ |
| Logistic Regression | 0.280  | 0.341    | 0.230     | 0.665  |
| Random Forest       | 0.781  | 0.818    | 0.938     | 0.725  |
| XGBoost             | 0.803  | 0.798    | 0.892     | 0.722  |
| GCN                 | 0.292  | 0.356    | 0.243     | 0.664  |
| GraphSAGE           | 0.253  | 0.420    | 0.334     | 0.563  |
| GAT                 | 0.307  | 0.409    | 0.293     | 0.677  |
| EvolveGCN           | 0.341  | 0.230    | 0.134     | 0.808  |
| TGN                 | 0.286  | 0.302    | 0.186     | 0.801  |


## Rodada 2 de Ajustes (Sem data leakage)

| Modelo              | PR-AUC | F1-score | Precision | Recall |
| ------------------- | ------ | -------- | --------- | ------ |
| Logistic Regression | 0.256  | 0.331    | 0.220     | 0.667  |
| Random Forest       | 0.782  | 0.817    | 0.939     | 0.723  |
| XGBoost             | 0.799  | 0.802    | 0.897     | 0.726  |
| GCN                 | 0.231  | 0.385    | 0.271     | 0.665  |
| GraphSAGE           | 0.326  | 0.449    | 0.373     | 0.561  |
| GAT                 | 0.298  | 0.435    | 0.324     | 0.665  |
| EvolveGCN           | 0.231  | 0.247    | 0.143     | 0.903  |
| TGN                 | 0.251  | 0.224    | 0.128     | 0.889  |


## Rodada 3 de Ajustes: class weights, early stopping e threshold tuning
### Antes de alterar e adicionar: best_val = -1 e best_state = model.state_dict()
===== RESULTADOS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.229360  0.305906   0.195297  0.705448
1        RandomForest  0.786939  0.786248   0.853896  0.728532
2             XGBoost  0.800324  0.776251   0.828272  0.730379

GCN
{'PR_AUC': 0.3061666664797773, 'F1': 0.443024494142705, 'Precision': 0.35986159169550175, 'Recall': 0.5761772853185596, 'model': 'GCN'}

===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.2785704698745555, 'F1': 0.43188064389477815, 'Precision': 0.3756830601092896, 'Recall': 0.5078485687903971, 'model': 'GraphSAGE'}

===== RESULTADOS GAT =====
{'PR_AUC': 0.29000854474451, 'F1': 0.46413199426111906, 'Precision': 0.37947214076246333, 'Recall': 0.5974145891043398, 'model': 'GAT'}

===== RESULTADOS EVOLVE GCN =====
{'PR_AUC': 0.2644962325122244, 'F1': 0.3357186669392762, 'Precision': 0.21559873949579833, 'Recall': 0.7580794090489381, 'model': 'EvolveGCN'}

===== RESULTADOS TGN =====
{'PR_AUC': 0.2981515813629563, 'F1': 0.3399638336347197, 'Precision': 0.2172573189522342, 'Recall': 0.7811634349030471, 'model': 'TGN'}

### Depois de alterar e adicionar: best_val = -1 e best_state = model.state_dict()

| Modelo    | PR-AUC | F1-score | Precision | Recall |
| --------- | ------ | -------- | --------- | ------ |
| GCN       | 0.332  | 0.460    | 0.416     | 0.513  |
| GraphSAGE | 0.267  | 0.434    | 0.327     | 0.645  |
| GAT       | 0.216  | 0.272    | 0.163     | 0.801  |
| EvolveGCN | 0.285  | 0.321    | 0.202     | 0.774  |
| TGN       | 0.174  | 0.297    | 0.185     | 0.759  |


