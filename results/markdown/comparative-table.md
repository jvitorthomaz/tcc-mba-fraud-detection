# Comparative Table

python3.10 -m venv venv                                                    
source venv/bin/activate

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
## rodada 1
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

## ----------------------------------------------------------------------------
## rodada 2
===== MELHORES PARÂMETROS XGBOOST =====
{'n_estimators': 503, 'max_depth': 8, 'learning_rate': 0.029700835355476407, 'subsample': 0.9998311111721847, 'colsample_bytree': 0.9985411440929729, 'min_child_weight': 2, 'gamma': 3.5950873897451108, 'reg_alpha': 3.5948409271845443, 'reg_lambda': 1.1544294613475528, 'scale_pos_weight': 7.892268052067664} | PR-AUC: 0.9966
===== MELHORES PARÂMETROS RANDOM FOREST =====
{'n_estimators': 358, 'max_depth': 14, 'min_samples_split': 9, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'class_weight': 'balanced'} | PR-AUC: 0.9922
===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====
{'C': 0.02281334680188677, 'class_weight': None}
{'C': 0.02281334680188677, 'class_weight': None} | PR-AUC: 0.6848

===== RESULTADOS FINAIS =====
MODELO                PR_AUC        F1  Precision    Recall  Threshold
XGBoost             0.804583  0.820075   0.972152  0.709141   0.900000
RandomForest        0.794613  0.819188   0.954545  0.717452   0.606122
LogisticRegression  0.415582  0.543618   0.540639  0.546630   0.720408
Ensemble            0.793992  0.802548   0.943820  0.698061   0.655102



===== MELHORES PARÂMETROS XGBOOST =====
{'n_estimators': 783, 'max_depth': 10, 'learning_rate': 0.015100332029457502, 'subsample': 0.999818919891117, 'colsample_bytree': 0.8481088824170445, 'min_child_weight': 5, 'gamma': 2.5550090067584246, 'reg_alpha': 2.643811149817767, 'reg_lambda': 1.1420684146941757, 'scale_pos_weight': 6.950609272915716}
{'n_estimators': 783, 'max_depth': 10, 'learning_rate': 0.015100332029457502, 'subsample': 0.999818919891117, 'colsample_bytree': 0.8481088824170445, 'min_child_weight': 5, 'gamma': 2.5550090067584246, 'reg_alpha': 2.643811149817767, 'reg_lambda': 1.1420684146941757, 'scale_pos_weight': 6.950609272915716} | PR-AUC: 0.9966
===== MELHORES PARÂMETROS RANDOM FOREST =====
{'n_estimators': 353, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'class_weight': 'balanced'}
{'n_estimators': 353, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'class_weight': 'balanced'} | PR-AUC: 0.9925
===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====
{'C': 0.022013290906413338, 'class_weight': None}
{'C': 0.022013290906413338, 'class_weight': None} | PR-AUC: 0.6848

===== RESULTADOS COMPLETOS =====
MODELO                      PR_AUC        F1  Precision    Recall  Threshold
XGBoost_Test              0.804379  0.745007   0.749533  0.740536   0.410204
RandomForest_Test         0.795215  0.813347   0.934132  0.720222   0.524490
LogisticRegression_Test   0.415876  0.507595   0.423886  0.632502   0.573469


===== RESULTADOS COMPLETOS =====
MODELO                      PR_AUC        F1  Precision    Recall  Threshold
XGBoost_Train             0.999994       NaN        NaN       NaN        NaN
XGBoost_Val               0.999969  0.999017   0.998035  1.000000        NaN
XGBoost_Test              0.804379  0.745007   0.749533  0.740536   0.410204
RandomForest_Train        0.999716       NaN        NaN       NaN        NaN
RandomForest_Val          0.999937  0.999017   0.998035  1.000000        NaN
RandomForest_Test         0.795215  0.813347   0.934132  0.720222   0.524490
LogisticRegression_Train  0.869161       NaN        NaN       NaN        NaN
LogisticRegression_Val    0.830665  0.818363   0.829960  0.807087        NaN
LogisticRegression_Test   0.415876  0.507595   0.423886  0.632502   0.573469



## --------------------------GRAFOS----------------------------

===== RESULTADOS FINAIS =====
MODELO                PR_AUC        F1  Precision    Recall
GCN_TEMPORAL        0.703548  0.651856   0.727906  0.590194
GraphSage_Temporal  0.703116  0.681925   0.734057  0.636706
GAT_Temporal        0.673279  0.599183   0.528310  0.692017
EvolveGCN           0.501491  0.510901   0.421676  0.648020
TGN_Temporal        


## GCN
===== MELHORES PARÂMETROS GCN =====
{'hidden_dim': 256, 'lr': 0.0028956146453530513, 'dropout': 0.5952925716514567, 'weight_decay': 0.00048273044132717376}

===== RESULTADOS GCN =====
{'PR_AUC': 0.759437367372766, 'F1': 0.6812004530011325, 'Precision': 0.6197836166924265, 'Recall': 0.7561282212445003, 'model': 'GCN_TEMPORAL'}

===== MELHORES PARÂMETROS GCN =====
{'hidden_dim': 256, 'lr': 0.0013218368325703402, 'dropout': 0.20550589596037555, 'weight_decay': 0.0009504533757483284}

===== RESULTADOS GCN  =====
{'PR_AUC': 0.7239495839842318, 'F1': 0.6320043103448276, 'Precision': 0.553041018387553, 'Recall': 0.7372721558768071, 'model': 'GCN'}

### Versão inicial das melhorias
===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 128, 'lr': 0.002888874568265193, 'dropout': 0.24752627452777007, 'weight_decay': 0.00014243209619628516}

===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7195272425004758, 'F1': 0.7267851156553805, 'Precision': 0.7787356321839081, 'Recall': 0.6813324952859836}

### Versão atualizada das melhorias

===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 128, 'lr': 0.0021183760207006386, 'dropout': 0.4594199931316609, 'weight_decay': 2.4685074147930517e-05}

===== RESULTADOS GRAPHSAGE FINAL =====
{'PR_AUC': 0.7425137608109611, 'F1': 0.6202433341962206, 'Precision': 0.5272887323943662, 'Recall': 0.7529855436832181}

## Versão final das melhorias (Versão "tunada")

===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 128, 'lr': 0.00030288408341720935, 'dropout': 0.46262289945729784, 'weight_decay': 1.2846498080446342e-05}

===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7274571347539416, 'F1': 0.5560239627246506, 'Precision': 0.42969821673525377, 'Recall': 0.7875549968573224}

===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 64, 'lr': 0.002072774282161724, 'dropout': 0.411266877681573, 'weight_decay': 8.024369116248995e-05}

===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7676852881511206, 'F1': 0.6716750139120757, 'Precision': 0.6025961058412381, 'Recall': 0.7586423632935261}

