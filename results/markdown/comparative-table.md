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
| Threshold tuning   | 3       | próxima    |
| XGBoost tuning     | 3       | próxima    |
| GraphSAGE tuning   | 2       | média      |
| Regularização GNN  | 1       | média      |
| Features temporais | 2       | depois     |
| Pipeline temporal  | 3       | opcional   |



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



## Leitura Técnica da Tabela (curta e forte)
- Melhor PR-AUC: XGBoost
- Melhor F1-score: Random Forest
- Melhor recall: EvolveGCN / Logistic / TGN
- Melhor equilíbrio geral: XGBoost

## Texto explicativo

A Tabela apresenta a comparação entre os modelos avaliados considerando as métricas PR-AUC, F1-score, precisão e recall. Observa-se que os modelos tabulares, especialmente XGBoost e Random Forest, apresentam desempenho superior em relação às demais abordagens, com destaque para o XGBoost, que obteve o maior PR-AUC e um bom equilíbrio entre precisão e recall. O Random Forest, por sua vez, apresentou o maior F1-score, impulsionado por uma precisão elevada.

Entre os modelos baseados em grafos, o GraphSAGE demonstrou o melhor desempenho geral, com o maior F1-score dentro desse grupo, enquanto o GAT apresentou maior PR-AUC, indicando melhor capacidade de ranqueamento das transações fraudulentas. O GCN apresentou desempenho mais limitado, sugerindo menor capacidade de captura de padrões complexos no grafo.

Os modelos temporais, EvolveGCN e TGN, apresentaram comportamento caracterizado por alto recall e baixa precisão, indicando tendência à superestimação da classe fraudulenta. Esse comportamento resulta em baixos valores de F1-score e PR-AUC, evidenciando limitações na configuração adotada para capturar dinâmicas temporais de forma eficaz.

## Insight

“Mesmo com maior complexidade, os modelos de grafos e temporais não superaram os modelos tabulares, o que evidencia que a qualidade da modelagem e da engenharia de features pode ser mais determinante do que a complexidade do modelo.”

## ENGENHARIA DE FEATURES PARA MELHORIAS DOS MODELOS???????


Entendo que GCN otimizada, hiperparâmetros ajustados e uso correto de regularização, normalização e profundidade, treino por timestep, avaliação mais contralada, uso de early stopping com validação controlada são fatores que são interessante de serem trabalhados durante o processo de ajuste e melhoria dos modelos. Agora, porque já não usamos features locais (transação), features agregadas (vizinhança), features temporais e engenharia específica para fraude desde o começo (fizemos apenas features tabulares originais)? Porque não fizemos "grafo até o tempo t → prever t+1"?

Acredito ser importate deixar claro essas escolhas e o motivos delas terem sido feitas em um paragrafo no texto do artigo