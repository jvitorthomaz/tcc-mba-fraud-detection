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







❯ python -m experiments.run_baseline                                        


===== SPLIT TEMPORAL =====
Train: 1 → 30 | 26905
Val: 31 → 34 | 2989
Test: 35+ | 16670

===== FEATURE ENGINEERING - TRAIN =====

===== GERANDO FEATURES DE GRAFO (SEM LEAKAGE) =====
Features de grafo adicionadas (sem leakage)

===== FEATURE ENGINEERING - VALIDATION =====

===== GERANDO FEATURES DE GRAFO (SEM LEAKAGE) =====
Features de grafo adicionadas (sem leakage)

===== FEATURE ENGINEERING - TEST =====

===== GERANDO FEATURES DE GRAFO (SEM LEAKAGE) =====
Features de grafo adicionadas (sem leakage)

===== TREINANDO MODELOS BASELINE =====
[I 2026-04-06 21:59:39,725] A new study created in memory with name: no-name-a4a0ce23-823c-4769-b023-21864af7fb46
[I 2026-04-06 22:00:09,171] Trial 0 finished with value: 0.9890260818297442 and parameters: {'n_estimators': 423, 'max_depth': 4, 'learning_rate': 0.06795265966445531, 'subsample': 0.7165708677907754, 'colsample_bytree': 0.8170709455931987, 'min_child_weight': 10, 'gamma': 2.573675765171407, 'reg_alpha': 2.5493705042872343, 'reg_lambda': 6.179378925074482}. Best is trial 0 with value: 0.9890260818297442.
[I 2026-04-06 22:00:57,006] Trial 1 finished with value: 0.9903772953654767 and parameters: {'n_estimators': 592, 'max_depth': 7, 'learning_rate': 0.08202832607584383, 'subsample': 0.8435105340471127, 'colsample_bytree': 0.891463584621198, 'min_child_weight': 9, 'gamma': 4.716536084658834, 'reg_alpha': 2.151604007338537, 'reg_lambda': 6.480694282865134}. Best is trial 1 with value: 0.9903772953654767.
[I 2026-04-06 22:01:13,948] Trial 2 finished with value: 0.9851840621647104 and parameters: {'n_estimators': 416, 'max_depth': 6, 'learning_rate': 0.08252608699366146, 'subsample': 0.6636621677194707, 'colsample_bytree': 0.8370511459844096, 'min_child_weight': 9, 'gamma': 3.0586283479452083, 'reg_alpha': 4.126033530005006, 'reg_lambda': 7.392728618792385}. Best is trial 1 with value: 0.9903772953654767.
[I 2026-04-06 22:01:46,309] Trial 3 finished with value: 0.9880076094660203 and parameters: {'n_estimators': 764, 'max_depth': 9, 'learning_rate': 0.05228034573373155, 'subsample': 0.8117065993999011, 'colsample_bytree': 0.6065810128159842, 'min_child_weight': 7, 'gamma': 3.738342858210309, 'reg_alpha': 4.831801744379084, 'reg_lambda': 6.965309809855558}. Best is trial 1 with value: 0.9903772953654767.
[I 2026-04-06 22:02:54,966] Trial 4 finished with value: 0.9916635341543609 and parameters: {'n_estimators': 665, 'max_depth': 5, 'learning_rate': 0.04426636361894012, 'subsample': 0.9871989324023583, 'colsample_bytree': 0.9823696831144477, 'min_child_weight': 7, 'gamma': 1.7744888343075966, 'reg_alpha': 1.608151531360691, 'reg_lambda': 9.340129736227134}. Best is trial 4 with value: 0.9916635341543609.
[I 2026-04-06 22:03:49,875] Trial 5 finished with value: 0.9865568526353468 and parameters: {'n_estimators': 700, 'max_depth': 7, 'learning_rate': 0.03445562227658637, 'subsample': 0.7072520649558343, 'colsample_bytree': 0.6005134135296591, 'min_child_weight': 7, 'gamma': 0.66164286385194, 'reg_alpha': 4.473896624952803, 'reg_lambda': 8.406899688349803}. Best is trial 4 with value: 0.9916635341543609.
[I 2026-04-06 22:05:42,997] Trial 6 finished with value: 0.9898653376797645 and parameters: {'n_estimators': 783, 'max_depth': 8, 'learning_rate': 0.028418037228207634, 'subsample': 0.7925633855620606, 'colsample_bytree': 0.8000308207632028, 'min_child_weight': 1, 'gamma': 0.47462272871020894, 'reg_alpha': 1.4946829391524123, 'reg_lambda': 9.881347631709316}. Best is trial 4 with value: 0.9916635341543609.
[I 2026-04-06 22:06:30,610] Trial 7 finished with value: 0.9894293749780423 and parameters: {'n_estimators': 704, 'max_depth': 10, 'learning_rate': 0.03600077998865209, 'subsample': 0.8138458964153518, 'colsample_bytree': 0.7099932266876146, 'min_child_weight': 10, 'gamma': 3.2090499207752012, 'reg_alpha': 2.7938091115693093, 'reg_lambda': 4.465732600344438}. Best is trial 4 with value: 0.9916635341543609.
[I 2026-04-06 22:06:57,349] Trial 8 finished with value: 0.9891866755118633 and parameters: {'n_estimators': 393, 'max_depth': 4, 'learning_rate': 0.09751858339184882, 'subsample': 0.6200796757084834, 'colsample_bytree': 0.720660993741423, 'min_child_weight': 7, 'gamma': 2.0803325686995535, 'reg_alpha': 3.9371244433513515, 'reg_lambda': 1.2397880269870492}. Best is trial 4 with value: 0.9916635341543609.
[I 2026-04-06 22:07:51,786] Trial 9 finished with value: 0.9900359437852151 and parameters: {'n_estimators': 461, 'max_depth': 8, 'learning_rate': 0.028748322884803106, 'subsample': 0.8187523020758195, 'colsample_bytree': 0.7025539042939704, 'min_child_weight': 3, 'gamma': 0.7597225404855035, 'reg_alpha': 2.3044950570017275, 'reg_lambda': 5.451869946892005}. Best is trial 4 with value: 0.9916635341543609.
[I 2026-04-06 22:09:37,685] Trial 10 finished with value: 0.9931052854071055 and parameters: {'n_estimators': 864, 'max_depth': 5, 'learning_rate': 0.013584244085915957, 'subsample': 0.9824385366942878, 'colsample_bytree': 0.9915011782145168, 'min_child_weight': 4, 'gamma': 1.6772917826005362, 'reg_alpha': 0.026478667338736717, 'reg_lambda': 2.87727471159093}. Best is trial 10 with value: 0.9931052854071055.
[I 2026-04-06 22:11:35,433] Trial 11 finished with value: 0.9925406076201865 and parameters: {'n_estimators': 895, 'max_depth': 5, 'learning_rate': 0.010969495928199458, 'subsample': 0.9854655284367282, 'colsample_bytree': 0.9950058555472128, 'min_child_weight': 4, 'gamma': 1.588368132392179, 'reg_alpha': 0.14196443381999133, 'reg_lambda': 2.8507165473385223}. Best is trial 10 with value: 0.9931052854071055.
[I 2026-04-06 22:13:23,905] Trial 12 finished with value: 0.9926999935275233 and parameters: {'n_estimators': 889, 'max_depth': 5, 'learning_rate': 0.012801028251888688, 'subsample': 0.9912027297124119, 'colsample_bytree': 0.999546297666609, 'min_child_weight': 4, 'gamma': 1.3005010260252836, 'reg_alpha': 0.31373591791926814, 'reg_lambda': 2.726274307278549}. Best is trial 10 with value: 0.9931052854071055.
[I 2026-04-06 22:15:05,809] Trial 13 finished with value: 0.9932433155033593 and parameters: {'n_estimators': 876, 'max_depth': 5, 'learning_rate': 0.01411675693392543, 'subsample': 0.9218380083554444, 'colsample_bytree': 0.9221010703939714, 'min_child_weight': 4, 'gamma': 1.2440177924984794, 'reg_alpha': 0.21897317917384046, 'reg_lambda': 3.148795390515028}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:16:48,390] Trial 14 finished with value: 0.9923709527799838 and parameters: {'n_estimators': 816, 'max_depth': 6, 'learning_rate': 0.01998797574508465, 'subsample': 0.9094890790304735, 'colsample_bytree': 0.9157993661364846, 'min_child_weight': 2, 'gamma': 0.048227429063559146, 'reg_alpha': 0.7700249730985574, 'reg_lambda': 3.7558821515185747}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:17:55,860] Trial 15 finished with value: 0.9921833845916546 and parameters: {'n_estimators': 552, 'max_depth': 6, 'learning_rate': 0.021003913396905653, 'subsample': 0.9186804184687614, 'colsample_bytree': 0.9226632736919761, 'min_child_weight': 5, 'gamma': 1.1836526700851722, 'reg_alpha': 0.8692996671836922, 'reg_lambda': 1.0753016520136438}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:18:22,813] Trial 16 finished with value: 0.9914189163433001 and parameters: {'n_estimators': 304, 'max_depth': 4, 'learning_rate': 0.04866708756966709, 'subsample': 0.9173879774639657, 'colsample_bytree': 0.9431993371051426, 'min_child_weight': 5, 'gamma': 2.0963706383614547, 'reg_alpha': 0.02034600078486519, 'reg_lambda': 2.4014092690736906}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:19:11,544] Trial 17 finished with value: 0.9913701658269696 and parameters: {'n_estimators': 839, 'max_depth': 5, 'learning_rate': 0.06405750525339757, 'subsample': 0.8711155663012914, 'colsample_bytree': 0.8664899352600499, 'min_child_weight': 3, 'gamma': 2.6203499813163176, 'reg_alpha': 3.2745411961938937, 'reg_lambda': 4.2228868930202825}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:21:08,676] Trial 18 finished with value: 0.9926269100478697 and parameters: {'n_estimators': 740, 'max_depth': 6, 'learning_rate': 0.02024780944139007, 'subsample': 0.9441794834863158, 'colsample_bytree': 0.9509120782308702, 'min_child_weight': 1, 'gamma': 1.0278132149221177, 'reg_alpha': 0.9450859215621912, 'reg_lambda': 5.200718621343485}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:22:09,628] Trial 19 finished with value: 0.991983793341282 and parameters: {'n_estimators': 846, 'max_depth': 4, 'learning_rate': 0.03768759548430991, 'subsample': 0.8695239060400766, 'colsample_bytree': 0.7636459007080447, 'min_child_weight': 6, 'gamma': 0.12807860576783292, 'reg_alpha': 1.4181146513157128, 'reg_lambda': 3.390707662817519}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:22:43,506] Trial 20 finished with value: 0.9909468115855956 and parameters: {'n_estimators': 652, 'max_depth': 8, 'learning_rate': 0.06244190069249196, 'subsample': 0.955084707955976, 'colsample_bytree': 0.870089686653022, 'min_child_weight': 3, 'gamma': 4.044886680778678, 'reg_alpha': 0.5211834912060291, 'reg_lambda': 1.8952432766256042}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:24:29,162] Trial 21 finished with value: 0.9925498872170841 and parameters: {'n_estimators': 881, 'max_depth': 5, 'learning_rate': 0.011274379336076045, 'subsample': 0.9919239874382605, 'colsample_bytree': 0.9977096097073287, 'min_child_weight': 4, 'gamma': 1.4856134816346525, 'reg_alpha': 0.35735274471814304, 'reg_lambda': 2.9388342375773764}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:26:12,838] Trial 22 finished with value: 0.9929912810130002 and parameters: {'n_estimators': 900, 'max_depth': 5, 'learning_rate': 0.012959004684846287, 'subsample': 0.9543077928305206, 'colsample_bytree': 0.9612010803020241, 'min_child_weight': 4, 'gamma': 1.235499534857247, 'reg_alpha': 0.004690492423809811, 'reg_lambda': 2.2348980934456986}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:27:42,160] Trial 23 finished with value: 0.9914019772150103 and parameters: {'n_estimators': 791, 'max_depth': 6, 'learning_rate': 0.026059054714778783, 'subsample': 0.9479870681103169, 'colsample_bytree': 0.9576518339294542, 'min_child_weight': 5, 'gamma': 2.2218106683921297, 'reg_alpha': 1.152871736605134, 'reg_lambda': 2.0096800916956075}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:29:09,939] Trial 24 finished with value: 0.993145310272642 and parameters: {'n_estimators': 816, 'max_depth': 5, 'learning_rate': 0.016037470027189867, 'subsample': 0.888907167576515, 'colsample_bytree': 0.8983114954595688, 'min_child_weight': 2, 'gamma': 1.7929979558981521, 'reg_alpha': 0.038268529605344534, 'reg_lambda': 4.496043938465916}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:30:26,381] Trial 25 finished with value: 0.9922675026785861 and parameters: {'n_estimators': 840, 'max_depth': 4, 'learning_rate': 0.020427961868005644, 'subsample': 0.8660268334453486, 'colsample_bytree': 0.9029621795089041, 'min_child_weight': 2, 'gamma': 1.9052776348567453, 'reg_alpha': 1.865133102815138, 'reg_lambda': 4.285305940641402}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:31:17,690] Trial 26 finished with value: 0.9913588808929454 and parameters: {'n_estimators': 747, 'max_depth': 7, 'learning_rate': 0.042021396044842854, 'subsample': 0.8971795046068481, 'colsample_bytree': 0.8325109493224807, 'min_child_weight': 2, 'gamma': 2.9921268515576793, 'reg_alpha': 0.6072727410289122, 'reg_lambda': 4.912413783836829}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:32:29,939] Trial 27 finished with value: 0.9912102376557986 and parameters: {'n_estimators': 815, 'max_depth': 6, 'learning_rate': 0.028814651482451343, 'subsample': 0.7805727700569742, 'colsample_bytree': 0.8771598101134837, 'min_child_weight': 3, 'gamma': 0.8651322695853967, 'reg_alpha': 1.0882746663792122, 'reg_lambda': 3.9038430524670225}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:33:29,735] Trial 28 finished with value: 0.991726035131339 and parameters: {'n_estimators': 544, 'max_depth': 5, 'learning_rate': 0.0184791973565023, 'subsample': 0.764701648428672, 'colsample_bytree': 0.9290769574436394, 'min_child_weight': 6, 'gamma': 2.4616777712417828, 'reg_alpha': 0.6142121816140768, 'reg_lambda': 3.3551578683041474}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:34:11,216] Trial 29 finished with value: 0.9885223413582714 and parameters: {'n_estimators': 856, 'max_depth': 4, 'learning_rate': 0.057532100972541286, 'subsample': 0.7488832198043192, 'colsample_bytree': 0.848606106533441, 'min_child_weight': 2, 'gamma': 2.594274156067088, 'reg_alpha': 2.9688687406372227, 'reg_lambda': 5.764826243914271}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:34:55,238] Trial 30 finished with value: 0.9929121337094937 and parameters: {'n_estimators': 713, 'max_depth': 4, 'learning_rate': 0.0785177005677904, 'subsample': 0.8831459741862182, 'colsample_bytree': 0.7820889713972219, 'min_child_weight': 1, 'gamma': 1.629100985157499, 'reg_alpha': 0.33444471092185724, 'reg_lambda': 4.809117082980614}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:36:42,854] Trial 31 finished with value: 0.9931305280747671 and parameters: {'n_estimators': 893, 'max_depth': 5, 'learning_rate': 0.014831592313779732, 'subsample': 0.9555765713260954, 'colsample_bytree': 0.9605289714849053, 'min_child_weight': 4, 'gamma': 1.3637758635704251, 'reg_alpha': 0.0952453405459239, 'reg_lambda': 1.673618673018649}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:38:32,695] Trial 32 finished with value: 0.9928598570890637 and parameters: {'n_estimators': 857, 'max_depth': 5, 'learning_rate': 0.01677653268973158, 'subsample': 0.9328847106376061, 'colsample_bytree': 0.9659716888079274, 'min_child_weight': 4, 'gamma': 0.4780997862934676, 'reg_alpha': 0.02835610138851576, 'reg_lambda': 1.5826184619915529}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:39:52,168] Trial 33 finished with value: 0.9906910627341736 and parameters: {'n_estimators': 805, 'max_depth': 6, 'learning_rate': 0.02407028948497012, 'subsample': 0.8442918693407502, 'colsample_bytree': 0.9122516510717333, 'min_child_weight': 5, 'gamma': 1.441488456259335, 'reg_alpha': 0.5127055484773846, 'reg_lambda': 3.3214626671194827}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:41:07,466] Trial 34 finished with value: 0.990530106899247 and parameters: {'n_estimators': 770, 'max_depth': 7, 'learning_rate': 0.0317466351409771, 'subsample': 0.9683103657456957, 'colsample_bytree': 0.8913219143761267, 'min_child_weight': 3, 'gamma': 1.908502354636168, 'reg_alpha': 1.2416810416918254, 'reg_lambda': 5.965336343734025}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:42:42,798] Trial 35 finished with value: 0.9913851681648083 and parameters: {'n_estimators': 859, 'max_depth': 5, 'learning_rate': 0.01507254728058486, 'subsample': 0.8427011767093211, 'colsample_bytree': 0.9756828723978959, 'min_child_weight': 6, 'gamma': 1.074060300302575, 'reg_alpha': 1.9242819700777487, 'reg_lambda': 1.5621498779210465}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:43:58,078] Trial 36 finished with value: 0.9930145322380977 and parameters: {'n_estimators': 820, 'max_depth': 4, 'learning_rate': 0.02422331159101919, 'subsample': 0.9681834957183234, 'colsample_bytree': 0.9347958898266138, 'min_child_weight': 4, 'gamma': 2.2834130130200876, 'reg_alpha': 0.29205996669024625, 'reg_lambda': 2.5487880363989412}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:45:07,220] Trial 37 finished with value: 0.9901526353476812 and parameters: {'n_estimators': 657, 'max_depth': 5, 'learning_rate': 0.010392769661803915, 'subsample': 0.928150497948251, 'colsample_bytree': 0.8948755463157492, 'min_child_weight': 8, 'gamma': 4.901809303317351, 'reg_alpha': 0.7130783580560994, 'reg_lambda': 6.527874891928579}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:46:05,057] Trial 38 finished with value: 0.9919849742105089 and parameters: {'n_estimators': 738, 'max_depth': 6, 'learning_rate': 0.040860831351648794, 'subsample': 0.8945869802752496, 'colsample_bytree': 0.9752309189901427, 'min_child_weight': 2, 'gamma': 2.8619766791670758, 'reg_alpha': 0.8942609682719527, 'reg_lambda': 3.0619869318759596}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:46:35,431] Trial 39 finished with value: 0.988121757008437 and parameters: {'n_estimators': 594, 'max_depth': 10, 'learning_rate': 0.07486515237993309, 'subsample': 0.9680600667132659, 'colsample_bytree': 0.6651271965881278, 'min_child_weight': 3, 'gamma': 1.7621701337694349, 'reg_alpha': 3.5252554008915276, 'reg_lambda': 3.701861889115585}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:47:21,596] Trial 40 finished with value: 0.9901593363291572 and parameters: {'n_estimators': 773, 'max_depth': 7, 'learning_rate': 0.08856879920054532, 'subsample': 0.8434953302756053, 'colsample_bytree': 0.8147373907124791, 'min_child_weight': 5, 'gamma': 0.8592144327564539, 'reg_alpha': 4.855243097797, 'reg_lambda': 8.325131452741449}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:48:30,168] Trial 41 finished with value: 0.9929695983552184 and parameters: {'n_estimators': 807, 'max_depth': 4, 'learning_rate': 0.02384601694589713, 'subsample': 0.9728681773326665, 'colsample_bytree': 0.933743945457218, 'min_child_weight': 4, 'gamma': 2.4054176125111275, 'reg_alpha': 0.27816512727290127, 'reg_lambda': 2.466380093483217}. Best is trial 13 with value: 0.9932433155033593.
[I 2026-04-06 22:49:38,593] Trial 42 finished with value: 0.9934090428764525 and parameters: {'n_estimators': 873, 'max_depth': 4, 'learning_rate': 0.03336560327329037, 'subsample': 0.9394775224229964, 'colsample_bytree': 0.9451558260166991, 'min_child_weight': 4, 'gamma': 3.5189041248460904, 'reg_alpha': 0.26827329261536387, 'reg_lambda': 1.7972901539967854}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:50:34,066] Trial 43 finished with value: 0.9929631329427534 and parameters: {'n_estimators': 872, 'max_depth': 4, 'learning_rate': 0.03275085570353304, 'subsample': 0.9995577577020983, 'colsample_bytree': 0.9788793724268773, 'min_child_weight': 3, 'gamma': 3.2964598043771742, 'reg_alpha': 0.4821619653125736, 'reg_lambda': 1.5321157168950557}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:52:09,081] Trial 44 finished with value: 0.9922973779770632 and parameters: {'n_estimators': 898, 'max_depth': 5, 'learning_rate': 0.016090103703725423, 'subsample': 0.9321399955436408, 'colsample_bytree': 0.9514586157416387, 'min_child_weight': 6, 'gamma': 4.326815341697327, 'reg_alpha': 0.22231726161584683, 'reg_lambda': 1.8911448580625638}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:54:19,833] Trial 45 finished with value: 0.992132562766501 and parameters: {'n_estimators': 876, 'max_depth': 9, 'learning_rate': 0.014740978491485814, 'subsample': 0.9024593489925135, 'colsample_bytree': 0.9139819327626592, 'min_child_weight': 4, 'gamma': 3.550439410509147, 'reg_alpha': 0.1425260656502965, 'reg_lambda': 1.095432121360655}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:55:02,645] Trial 46 finished with value: 0.991521210167543 and parameters: {'n_estimators': 831, 'max_depth': 5, 'learning_rate': 0.047642661615833456, 'subsample': 0.6989274487695676, 'colsample_bytree': 0.9861826967601331, 'min_child_weight': 5, 'gamma': 1.424263013999173, 'reg_alpha': 2.5245575171555314, 'reg_lambda': 3.956828871737935}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:55:58,954] Trial 47 finished with value: 0.986891433079241 and parameters: {'n_estimators': 783, 'max_depth': 4, 'learning_rate': 0.010153073853621106, 'subsample': 0.6010678746341195, 'colsample_bytree': 0.852025419981621, 'min_child_weight': 10, 'gamma': 4.448328061681627, 'reg_alpha': 0.77338178335505, 'reg_lambda': 2.112573199549563}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:56:52,789] Trial 48 finished with value: 0.9931427843102048 and parameters: {'n_estimators': 494, 'max_depth': 5, 'learning_rate': 0.030671560956360727, 'subsample': 0.9399763968808202, 'colsample_bytree': 0.9484301972935433, 'min_child_weight': 3, 'gamma': 1.9282395209210788, 'reg_alpha': 1.409245218143299, 'reg_lambda': 2.7938163370307896}. Best is trial 42 with value: 0.9934090428764525.
[I 2026-04-06 22:57:59,544] Trial 49 finished with value: 0.992269476491916 and parameters: {'n_estimators': 541, 'max_depth': 6, 'learning_rate': 0.036041358380941135, 'subsample': 0.9191108846764916, 'colsample_bytree': 0.8891900179123418, 'min_child_weight': 2, 'gamma': 2.003851912729235, 'reg_alpha': 1.526729604598394, 'reg_lambda': 1.46372882511192}. Best is trial 42 with value: 0.9934090428764525.

===== MELHORES PARÂMETROS XGBOOST =====
{'n_estimators': 873, 'max_depth': 4, 'learning_rate': 0.03336560327329037, 'subsample': 0.9394775224229964, 'colsample_bytree': 0.9451558260166991, 'min_child_weight': 4, 'gamma': 3.5189041248460904, 'reg_alpha': 0.26827329261536387, 'reg_lambda': 1.7972901539967854}
[I 2026-04-06 22:59:23,574] A new study created in memory with name: no-name-d1a94228-640b-4705-9994-730882d8ce0b
[I 2026-04-06 22:59:28,488] Trial 0 finished with value: 0.9823719382548181 and parameters: {'n_estimators': 345, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 0 with value: 0.9823719382548181.
[I 2026-04-06 22:59:35,047] Trial 1 finished with value: 0.9896688476040063 and parameters: {'n_estimators': 269, 'max_depth': 13, 'min_samples_split': 3, 'min_samples_leaf': 4}. Best is trial 1 with value: 0.9896688476040063.
[I 2026-04-06 22:59:41,416] Trial 2 finished with value: 0.9902584021463426 and parameters: {'n_estimators': 266, 'max_depth': 12, 'min_samples_split': 9, 'min_samples_leaf': 1}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 22:59:46,535] Trial 3 finished with value: 0.9743944811989894 and parameters: {'n_estimators': 445, 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 5}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 22:59:50,499] Trial 4 finished with value: 0.9856664179829799 and parameters: {'n_estimators': 249, 'max_depth': 6, 'min_samples_split': 6, 'min_samples_leaf': 2}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 22:59:58,189] Trial 5 finished with value: 0.9902310971292856 and parameters: {'n_estimators': 308, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 4}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:01,359] Trial 6 finished with value: 0.9756631615662397 and parameters: {'n_estimators': 263, 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 5}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:10,074] Trial 7 finished with value: 0.9874245351284218 and parameters: {'n_estimators': 477, 'max_depth': 7, 'min_samples_split': 8, 'min_samples_leaf': 2}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:11,870] Trial 8 finished with value: 0.9745146843936346 and parameters: {'n_estimators': 140, 'max_depth': 4, 'min_samples_split': 4, 'min_samples_leaf': 5}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:14,787] Trial 9 finished with value: 0.9888947957210505 and parameters: {'n_estimators': 129, 'max_depth': 9, 'min_samples_split': 7, 'min_samples_leaf': 5}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:24,072] Trial 10 finished with value: 0.990246851172628 and parameters: {'n_estimators': 388, 'max_depth': 12, 'min_samples_split': 10, 'min_samples_leaf': 1}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:33,239] Trial 11 finished with value: 0.9902118313810208 and parameters: {'n_estimators': 390, 'max_depth': 12, 'min_samples_split': 10, 'min_samples_leaf': 1}. Best is trial 2 with value: 0.9902584021463426.
[I 2026-04-06 23:00:37,890] Trial 12 finished with value: 0.9906195162455111 and parameters: {'n_estimators': 200, 'max_depth': 11, 'min_samples_split': 9, 'min_samples_leaf': 2}. Best is trial 12 with value: 0.9906195162455111.
[I 2026-04-06 23:00:42,630] Trial 13 finished with value: 0.989812725305047 and parameters: {'n_estimators': 213, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2}. Best is trial 12 with value: 0.9906195162455111.
[I 2026-04-06 23:00:46,571] Trial 14 finished with value: 0.989913581960434 and parameters: {'n_estimators': 174, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3}. Best is trial 12 with value: 0.9906195162455111.
[I 2026-04-06 23:00:51,684] Trial 15 finished with value: 0.9907720031595086 and parameters: {'n_estimators': 196, 'max_depth': 14, 'min_samples_split': 9, 'min_samples_leaf': 2}. Best is trial 15 with value: 0.9907720031595086.
[I 2026-04-06 23:00:56,739] Trial 16 finished with value: 0.9904561000248374 and parameters: {'n_estimators': 194, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 3}. Best is trial 15 with value: 0.9907720031595086.
[I 2026-04-06 23:00:59,829] Trial 17 finished with value: 0.9905033050654841 and parameters: {'n_estimators': 110, 'max_depth': 14, 'min_samples_split': 9, 'min_samples_leaf': 2}. Best is trial 15 with value: 0.9907720031595086.
[I 2026-04-06 23:01:03,357] Trial 18 finished with value: 0.9880960426538472 and parameters: {'n_estimators': 174, 'max_depth': 8, 'min_samples_split': 7, 'min_samples_leaf': 3}. Best is trial 15 with value: 0.9907720031595086.
[I 2026-04-06 23:01:09,058] Trial 19 finished with value: 0.990369844204667 and parameters: {'n_estimators': 232, 'max_depth': 11, 'min_samples_split': 9, 'min_samples_leaf': 3}. Best is trial 15 with value: 0.9907720031595086.
[I 2026-04-06 23:01:17,425] Trial 20 finished with value: 0.9899555596040452 and parameters: {'n_estimators': 320, 'max_depth': 14, 'min_samples_split': 7, 'min_samples_leaf': 2}. Best is trial 15 with value: 0.9907720031595086.
[I 2026-04-06 23:01:20,250] Trial 21 finished with value: 0.9907983149074384 and parameters: {'n_estimators': 101, 'max_depth': 14, 'min_samples_split': 9, 'min_samples_leaf': 2}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:24,382] Trial 22 finished with value: 0.9894625047401792 and parameters: {'n_estimators': 156, 'max_depth': 13, 'min_samples_split': 9, 'min_samples_leaf': 2}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:27,250] Trial 23 finished with value: 0.990112535858741 and parameters: {'n_estimators': 107, 'max_depth': 14, 'min_samples_split': 8, 'min_samples_leaf': 2}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:32,278] Trial 24 finished with value: 0.9891428671553625 and parameters: {'n_estimators': 211, 'max_depth': 11, 'min_samples_split': 10, 'min_samples_leaf': 3}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:36,085] Trial 25 finished with value: 0.9892189468972359 and parameters: {'n_estimators': 153, 'max_depth': 13, 'min_samples_split': 9, 'min_samples_leaf': 1}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:38,745] Trial 26 finished with value: 0.9902607743246256 and parameters: {'n_estimators': 102, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 4}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:43,774] Trial 27 finished with value: 0.9901644851475568 and parameters: {'n_estimators': 198, 'max_depth': 11, 'min_samples_split': 8, 'min_samples_leaf': 2}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:47,624] Trial 28 finished with value: 0.9892048615010779 and parameters: {'n_estimators': 177, 'max_depth': 9, 'min_samples_split': 7, 'min_samples_leaf': 2}. Best is trial 21 with value: 0.9907983149074384.
[I 2026-04-06 23:01:56,075] Trial 29 finished with value: 0.9902266824383609 and parameters: {'n_estimators': 332, 'max_depth': 14, 'min_samples_split': 10, 'min_samples_leaf': 1}. Best is trial 21 with value: 0.9907983149074384.

===== MELHORES PARÂMETROS RANDOM FOREST =====
{'n_estimators': 101, 'max_depth': 14, 'min_samples_split': 9, 'min_samples_leaf': 2}
[I 2026-04-06 23:01:59,090] A new study created in memory with name: no-name-0b85f5e4-7a30-4543-ae89-c599ef9e3001
[I 2026-04-06 23:02:05,792] Trial 0 finished with value: 0.641795553201762 and parameters: {'C': 0.23764953361182045}. Best is trial 0 with value: 0.641795553201762.
[I 2026-04-06 23:02:07,984] Trial 1 finished with value: 0.6817187681398953 and parameters: {'C': 0.004865053809075721}. Best is trial 1 with value: 0.6817187681398953.
[I 2026-04-06 23:02:16,542] Trial 2 finished with value: 0.6416150617660901 and parameters: {'C': 0.8673780771826822}. Best is trial 1 with value: 0.6817187681398953.
[I 2026-04-06 23:02:37,456] Trial 3 finished with value: 0.6887714439892565 and parameters: {'C': 9.509238146019849}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:02:42,641] Trial 4 finished with value: 0.6409929359837913 and parameters: {'C': 0.2592780487321025}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:02:45,761] Trial 5 finished with value: 0.6825582165687806 and parameters: {'C': 0.028129560601801138}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:03:04,501] Trial 6 finished with value: 0.6866292354458472 and parameters: {'C': 8.949875188549123}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:03:08,893] Trial 7 finished with value: 0.6474030576807185 and parameters: {'C': 0.15033626354488425}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:03:10,781] Trial 8 finished with value: 0.6602870936005849 and parameters: {'C': 0.0014965408851753436}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:03:15,298] Trial 9 finished with value: 0.648921850296033 and parameters: {'C': 0.13743988261630155}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:03:30,782] Trial 10 finished with value: 0.6707361331404926 and parameters: {'C': 5.335932727137444}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:03:49,544] Trial 11 finished with value: 0.6864215008811193 and parameters: {'C': 8.854274042648795}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:04:01,255] Trial 12 finished with value: 0.6532442283336586 and parameters: {'C': 2.734995884480328}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:04:10,386] Trial 13 finished with value: 0.6447260272503298 and parameters: {'C': 1.593133203633949}. Best is trial 3 with value: 0.6887714439892565.
[I 2026-04-06 23:04:30,256] Trial 14 finished with value: 0.6896056493588862 and parameters: {'C': 9.736591683687406}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:04:39,014] Trial 15 finished with value: 0.6413488813662425 and parameters: {'C': 0.8370330403027965}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:04:50,588] Trial 16 finished with value: 0.6495678560397832 and parameters: {'C': 2.2771303479976504}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:04:53,175] Trial 17 finished with value: 0.680160143260163 and parameters: {'C': 0.0322872634718569}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:04:59,679] Trial 18 finished with value: 0.6389147058052018 and parameters: {'C': 0.5961811681691096}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:05:02,438] Trial 19 finished with value: 0.6756358355945855 and parameters: {'C': 0.042266969410247615}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:05:16,630] Trial 20 finished with value: 0.6646967768788948 and parameters: {'C': 4.326968071903245}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:05:36,025] Trial 21 finished with value: 0.6887662198210649 and parameters: {'C': 9.593716187792731}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:05:54,807] Trial 22 finished with value: 0.6873694541798506 and parameters: {'C': 9.101460139922112}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:06:06,638] Trial 23 finished with value: 0.6547799712272558 and parameters: {'C': 2.939311031342887}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:06:15,345] Trial 24 finished with value: 0.6445886722384956 and parameters: {'C': 1.2160451728337232}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:06:28,553] Trial 25 finished with value: 0.6628721339633774 and parameters: {'C': 4.073907002064805}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:06:30,254] Trial 26 finished with value: 0.6831283665756556 and parameters: {'C': 0.011705966062443314}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:06:45,957] Trial 27 finished with value: 0.6727480157040953 and parameters: {'C': 5.656031965283484}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:06:51,867] Trial 28 finished with value: 0.6378779665460861 and parameters: {'C': 0.502448799478657}. Best is trial 14 with value: 0.6896056493588862.
[I 2026-04-06 23:07:01,876] Trial 29 finished with value: 0.6465253535069231 and parameters: {'C': 1.85595106140657}. Best is trial 14 with value: 0.6896056493588862.

===== MELHORES PARÂMETROS LOGISTIC REGRESSION =====
{'C': 9.736591683687406}

===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall
XGBoost             0.807826  0.820704   0.972187  0.710065
RandomForest        0.793882  0.826341   0.972500  0.718375
LogisticRegression  0.318930  0.476551   0.403587  0.581717
Ensemble            0.797656  0.820350   0.962687  0.714681

===== RESULTADOS GCN TEMPORAL =====
{'PR_AUC': 0.7035481785950182, 'F1': 0.6518569940992711, 'Precision': 0.727906976744186, 'Recall': 0.5901948460087995, 'model': 'GCN_TEMPORAL'}

===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall
LogisticRegression  0.318930  0.476551   0.403587  0.581717
RandomForest        0.793882  0.826341   0.972500  0.718375
XGBoost             0.807826  0.820704   0.972187  0.710065
Ensemble            0.797656  0.820350   0.962687  0.714681
GCN_TEMPORAL        0.703548  0.651856   0.727906  0.590194


O que melhorar GCN:
- Ajuste fino (rápido ganho)
- tuning de threshold focado em recall
- aumentar epochs por step (20 → 30)
- testar hidden_dim (64 → 128)


===== RESULTADOS GRAPHSAGE TEMPORAL =====
{'PR_AUC': 0.7031165069186897, 'F1': 0.6819252776842815, 'Precision': 0.7340579710144928, 'Recall': 0.6367064739157763, 'model': 'GraphSAGE_TEMPORAL'}

===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall
LogisticRegression  0.318930  0.476551   0.403587  0.581717
RandomForest        0.793882  0.826341   0.972500  0.718375
XGBoost             0.807826  0.820704   0.972187  0.710065
Ensemble            0.797656  0.820350   0.962687  0.714681
GCN_TEMPORAL        0.703548  0.651856   0.727906  0.590194
GraphSage_Temporal  0.703116  0.681925   0.734057  0.636706


===== RESULTADOS GAT =====
{'PR_AUC': 0.28137133194306113, 'F1': 0.40988467874794066, 'Precision': 0.31864754098360654, 'Recall': 0.5743305632502308, 'model': 'GAT'}
===== RESULTADOS GAT TEMPORAL =====
{'PR_AUC': 0.6732799206023171, 'F1': 0.5991836734693877, 'Precision': 0.5283109404990403, 'Recall': 0.6920175989943432, 'model': 'GAT_TEMPORAL'}


===== RESULTADOS FINAIS =====
                      PR_AUC        F1  Precision    Recall
LogisticRegression  0.318930  0.476551   0.403587  0.581717
RandomForest        0.793882  0.826341   0.972500  0.718375
XGBoost             0.807826  0.820704   0.972187  0.710065
Ensemble            0.797656  0.820350   0.962687  0.714681
GCN_TEMPORAL        0.703548  0.651856   0.727906  0.590194
GraphSage_Temporal  0.703116  0.681925   0.734057  0.636706
GAT_Temporal        0.673279  0.599183   0.528310  0.692017




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
