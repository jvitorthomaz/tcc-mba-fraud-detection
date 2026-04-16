# Comparative Table

python3.10 -m venv venv                                                    
source venv/bin/activate

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


===== RESULTADOS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.227072  0.308877   0.197871  0.703601
1        RandomForest  0.781757  0.761490   0.799797  0.726685
2             XGBoost  0.804513  0.774006   0.815117  0.736842



===== RESULTADOS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.238691  0.297553   0.188018  0.712835
1        RandomForest  0.785386  0.750831   0.772461  0.730379
2             XGBoost  0.799893  0.760859   0.787549  0.735919



===== RESULTADOS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.226824  0.308910   0.197971  0.702678
1        RandomForest  0.788800  0.758258   0.787276  0.731302
2             XGBoost  0.797570  0.782007   0.841489  0.730379


===== RESULTADOS FINAIS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.238351  0.312201   0.196955  0.752539
1        RandomForest  0.784590  0.750478   0.777998  0.724838
2             XGBoost  0.794278  0.746818   0.763006  0.731302
3            Ensemble  0.791499  0.802728   0.929526  0.706371


===== RESULTADOS FINAIS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.236108  0.299443   0.189078  0.719298
1        RandomForest  0.789679  0.808050   0.915789  0.722992
2             XGBoost  0.798488  0.794949   0.877369  0.726685
3     Ensemble_RF_XGB  0.797156  0.822961   0.982074  0.708218


===== RESULTADOS FINAIS =====
                model    PR_AUC        F1  Precision    Recall
0  LogisticRegression  0.236108  0.299443   0.189078  0.719298
1        RandomForest  0.789679  0.808050   0.915789  0.722992
2             XGBoost  0.798488  0.794949   0.877369  0.726685
3     Ensemble_RF_XGB  0.797156  0.822961   0.982074  0.708218



## -------------------------------------------------------------------


===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====
{'C': 9.736591683687406}

===== MELHORES PARÂMETROS RANDOM FOREST =====
{'n_estimators': 101, 'max_depth': 14, 'min_samples_split': 9, 'min_samples_leaf': 2}

===== MELHORES PARÂMETROS XGBOOST =====
{'n_estimators': 873, 'max_depth': 4, 'learning_rate': 0.03336560327329037, 'subsample': 0.9394775224229964, 'colsample_bytree': 0.9451558260166991, 'min_child_weight': 4, 'gamma': 3.5189041248460904, 'reg_alpha': 0.26827329261536387, 'reg_lambda': 1.7972901539967854}

===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall
XGBoost             0.807826  0.820704   0.972187  0.710065
RandomForest        0.793882  0.826341   0.972500  0.718375
LogisticRegression  0.318930  0.476551   0.403587  0.581717
Ensemble            0.797656  0.820350   0.962687  0.714681

===== RESULTADOS GCN TEMPORAL =====
{'PR_AUC': 0.7035481785950182, 'F1': 0.6518569940992711, 'Precision': 0.727906976744186, 'Recall': 0.5901948460087995, 'model': 'GCN_TEMPORAL'}

O que melhorar GCN:
- Ajuste fino (rápido ganho)
- tuning de threshold focado em recall
- aumentar epochs por step (20 → 30)
- testar hidden_dim (64 → 128)


===== RESULTADOS GRAPHSAGE TEMPORAL =====
{'PR_AUC': 0.7031165069186897, 'F1': 0.6819252776842815, 'Precision': 0.7340579710144928, 'Recall': 0.6367064739157763, 'model': 'GraphSAGE_TEMPORAL'}

===== RESULTADOS GAT =====
{'PR_AUC': 0.28137133194306113, 'F1': 0.40988467874794066, 'Precision': 0.31864754098360654, 'Recall': 0.5743305632502308, 'model': 'GAT'}
===== RESULTADOS GAT TEMPORAL =====
{'PR_AUC': 0.6732799206023171, 'F1': 0.5991836734693877, 'Precision': 0.5283109404990403, 'Recall': 0.6920175989943432, 'model': 'GAT_TEMPORAL'}

===== RESULTADOS EVOLVE GCN =====
{'PR_AUC': 0.2619594247257695, 'F1': 0.3362978283350569, 'Precision': 0.21668443496801706, 'Recall': 0.7506925207756233, 'model': 'EvolveGCN'}
===== RESULTADOS EVOLVE GCN TEMPORAL =====
{'PR_AUC': 0.5014913151769682, 'F1': 0.5109018830525273, 'Precision': 0.42167689161554195, 'Recall': 0.6480201131363922, 'model': 'EvolveGCN_TEMPORAL_REAL'}

===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall
LogisticRegression  0.318930  0.476551   0.403587  0.581717
RandomForest        0.793882  0.826341   0.972500  0.718375
XGBoost             0.807826  0.820704   0.972187  0.710065
Ensemble            0.797656  0.820350   0.962687  0.714681
GCN_TEMPORAL        0.703548  0.651856   0.727906  0.590194
GraphSage_Temporal  0.703116  0.681925   0.734057  0.636706
GAT_Temporal        0.673279  0.599183   0.528310  0.692017
EvolveGCN           0.501491  0.510901   0.421676  0.648020



===== RESULTADOS TGN =====
{'PR_AUC': 0.2987778533353763, 'F1': 0.3580143037442154, 'Precision': 0.2318169436120948, 'Recall': 0.7857802400738689, 'model': 'TGN'}
===== RESULTADOS TGN TEMPORAL =====
{'PR_AUC': 0.06358254709221058, 'F1': 0.15139061454649716, 'Precision': 0.08192397207137316, 'Recall': 0.9956002514142049, 'model': 'TGN_TEMPORAL_REAL'}


===== RESULTADOS FINAIS =====
MODELO                PR_AUC        F1  Precision    Recall
LogisticRegression  0.318930  0.476551   0.403587  0.581717
RandomForest        0.793882  0.826341   0.972500  0.718375
XGBoost             0.807826  0.820704   0.972187  0.710065
Ensemble            0.797656  0.820350   0.962687  0.714681
GCN_TEMPORAL        0.703548  0.651856   0.727906  0.590194
GraphSage_Temporal  0.703116  0.681925   0.734057  0.636706
GAT_Temporal        0.673279  0.599183   0.528310  0.692017
EvolveGCN           0.501491  0.510901   0.421676  0.648020
TGN_Temporal        


## ----------------------------------------------------------------------------

===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====
{'C': 9.899964602858315, 'class_weight': None}
===== MELHORES PARÂMETROS RANDOM FOREST =====
{'n_estimators': 103, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'class_weight': 'balanced'}
===== MELHORES PARÂMETROS XGBOOST =====
{'n_estimators': 493, 'max_depth': 9, 'learning_rate': 0.03375537723376297, 'subsample': 0.7551100513887657, 'colsample_bytree': 0.8845136315827655, 'min_child_weight': 9, 'gamma': 1.1566537987443726, 'reg_alpha': 0.8795899916963011, 'reg_lambda': 1.5120031865963335, 'scale_pos_weight': 7.520091825863948}

===== RESULTADOS FINAIS =====
MODELO                PR_AUC        F1  Precision    Recall
XGBoost             0.805814  0.822529   0.968711  0.714681
RandomForest        0.795575  0.820730   0.960396  0.716528
LogisticRegression  0.318542  0.477612   0.407843  0.576177
Ensemble            0.795810  0.805263   0.936353  0.706371