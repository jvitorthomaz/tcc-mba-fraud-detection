# Metrics


## Tabulares - Threshold fixo

### 1
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


### 2 -> MAIS CORRETO METODOLOGICA ENTRE AS 3 PRIMEIRAS
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

### 3
===== MELHORES PARÂMETROS XGBOOST =====
{'n_estimators': 686, 'max_depth': 10, 'learning_rate': 0.015871642615770946, 'subsample': 0.9841014677771607, 'colsample_bytree': 0.6597408903772877, 'min_child_weight': 4, 'gamma': 4.54524368065516, 'reg_alpha': 0.589737282513381, 'reg_lambda': 1.0519176571430018, 'scale_pos_weight': 6.626249492461672}
{'n_estimators': 686, 'max_depth': 10, 'learning_rate': 0.015871642615770946, 'subsample': 0.9841014677771607, 'colsample_bytree': 0.6597408903772877, 'min_child_weight': 4, 'gamma': 4.54524368065516, 'reg_alpha': 0.589737282513381, 'reg_lambda': 1.0519176571430018, 'scale_pos_weight': 6.626249492461672} | PR-AUC: 0.9965
===== MELHORES PARÂMETROS RANDOM FOREST =====
{'n_estimators': 208, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced'}
{'n_estimators': 208, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced'} | PR-AUC: 0.9931
===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====
{'C': 9.9443126549784, 'class_weight': None}
{'C': 9.9443126549784, 'class_weight': None} | PR-AUC: 0.6902

===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall  Threshold
XGBoost             0.804389  0.769679   0.812308  0.731302        0.5
RandomForest        0.794338  0.810950   0.920281  0.724838        0.5
LogisticRegression  0.318390  0.457180   0.353119  0.648199        0.5



## Tabulares - Threshold Dinamico







## GCN - Threshold fixo

### 1
===== MELHORES PARÂMETROS GCN =====
{'hidden_dim': 256, 'lr': 0.0013218368325703402, 'dropout': 0.20550589596037555, 'weight_decay': 0.0009504533757483284}
===== RESULTADOS GCN  =====
{'PR_AUC': 0.7239495839842318, 'F1': 0.6320043103448276, 'Precision': 0.553041018387553, 'Recall': 0.7372721558768071, 'model': 'GCN'}

### 2
===== MELHORES PARÂMETROS GCN =====
{'hidden_dim': 128, 'lr': 0.003880772288316006, 'dropout': 0.31940615651860854, 'weight_decay': 0.00014113107862990596}
{'PR_AUC': 0.7116981308323147, 'F1': 0.6263910969793322, 'Precision': 0.5414567109482363, 'Recall': 0.742928975487115}



## GCN - Threshold Misto (dinamico e fixo)
===== MELHORES PARÂMETROS GCN =====
{'hidden_dim': 256, 'lr': 0.00288940217055051, 'dropout': 0.48172909715991824, 'weight_decay': 0.0009524555756215927}
{'PR_AUC': 0.6924660909751961, 'F1': 0.5732246798603027, 'Precision': 0.4552514792899408, 'Recall': 0.7737272155876807}


## GCN - Threshold Dinamico
===== MELHORES PARÂMETROS GCN =====
{'hidden_dim': 256, 'lr': 0.002870463254935626, 'dropout': 0.2941425191715136, 'weight_decay': 0.0006286062704380455}
{'PR_AUC': 0.7036711881535813, 'F1': 0.5832865956253506, 'Precision': 0.5265822784810127, 'Recall': 0.6536769327467001}





## GraphSAGE - Threshold fixo

### 1
===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 64, 'lr': 0.002072774282161724, 'dropout': 0.411266877681573, 'weight_decay': 8.024369116248995e-05}
===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7676852881511206, 'F1': 0.6716750139120757, 'Precision': 0.6025961058412381, 'Recall': 0.7586423632935261}

### 2
===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 128, 'lr': 0.0006241597602084456, 'dropout': 0.30453065508143384, 'weight_decay': 9.589922489396406e-05}
===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7534813074090013, 'F1': 0.6407557354925777, 'Precision': 0.5614947965941344, 'Recall': 0.7460716530483973}

### 3 
===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 256, 'lr': 0.0014125599736463902, 'dropout': 0.39453014437579553, 'weight_decay': 0.00020981380006508205}
===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7236522216281148, 'F1': 0.60272614622057, 'Precision': 0.49754500818330605, 'Recall': 0.764299182903834, 'model': 'GraphSAGE'}



## GraphSAGE - Threshold Misto (dinamico e fixo)
===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 128, 'lr': 0.0029494779986227455, 'dropout': 0.24730997153143225, 'weight_decay': 0.000762439668256296}
===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.7420301618509241, 'F1': 0.6449907480835315, 'Precision': 0.5565693430656934, 'Recall': 0.7668133249528598, 'model': 'GraphSAGE'}



## GraphSAGE - Threshold Dinamico
===== MELHORES PARÂMETROS GRAPHSAGE =====
{'hidden_dim': 64, 'lr': 0.004763635885421962, 'dropout': 0.4171118449097817, 'weight_decay': 0.00033967257922120123}
===== RESULTADOS GRAPHSAGE =====
{'PR_AUC': 0.75547421961677, 'F1': 0.6670537010159651, 'Precision': 0.6197411003236246, 'Recall': 0.7221873035826524, 'model': 'GraphSAGE'}



## GAT - Threshold fixo

### 1
===== MELHORES PARÂMETROS GAT =====
{'hidden_dim': 64, 'heads': 4, 'lr': 0.0040725849067926425, 'dropout': 0.36030567465597596, 'weight_decay': 2.3070806808554974e-05}
===== RESULTADOS GAT  =====
{'PR_AUC': 0.6756422896524013, 'F1': 0.5130542892664732, 'Precision': 0.38268933539412675, 'Recall': 0.7781269641734758}

### 2
===== MELHORES PARÂMETROS GAT =====
{'hidden_dim': 64, 'heads': 2, 'lr': 0.0035102040413411925, 'dropout': 0.22318315744020972, 'weight_decay': 1.0521903328281659e-06}

===== RESULTADOS GAT =====
{'PR_AUC': 0.6886808189736242, 'F1': 0.6403557531962202, 'Precision': 0.5739910313901345, 'Recall': 0.7240729101194218}



## GAT - Threshold Misto (dinamico e fixo)
===== MELHORES PARÂMETROS GAT =====
{'hidden_dim': 128, 'heads': 2, 'lr': 0.001339299352664218, 'dropout': 0.2281938488603404, 'weight_decay': 6.151948992541052e-06}

===== RESULTADOS GAT =====
{'PR_AUC': 0.6708074017642807, 'F1': 0.6183333333333333, 'Precision': 0.554006968641115, 'Recall': 0.6995600251414205}



## GAT - Threshold Dinamico
