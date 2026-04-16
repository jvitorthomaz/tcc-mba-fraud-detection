Tabela X – Hiperparâmetros do modelo XGBoost

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Modelo               | XGBClassifier |
| Número de árvores    | 100 |
| Learning rate        | 0.1 |
| Max depth            | 6 |
| Subsample            | 1.0 |
| Colsample bytree     | 1.0 |
| Gamma                | 0 |
| Scale pos weight     | Balanceado (neg/pos) |
| Objective            | binary:logistic |
| Eval metric          | logloss |
| Random state         | 42 |
| Estratégia temporal  | Split fixo (train/val/test) |

Tabela X – Hiperparâmetros do modelo Random Forest

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Modelo               | RandomForestClassifier |
| Número de árvores    | 100 |
| Profundidade máxima  | None |
| Min samples split    | 2 |
| Min samples leaf     | 1 |
| Max features         | sqrt |
| Class weight         | balanceado |
| Random state         | 42 |
| Estratégia temporal  | Split fixo (train/val/test) |

Tabela X – Hiperparâmetros do modelo Regressão Logística

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Modelo               | LogisticRegression |
| Solver               | lbfgs |
| Regularização        | L2 |
| C (inverso da reg.)  | 1.0 |
| Class weight         | balanceado |
| Máx. iterações       | 1000 |
| Normalização         | StandardScaler |
| Estratégia temporal  | Split fixo (train/val/test) |



Tabela X – Hiperparâmetros do modelo GCN Temporal

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Hidden dimensions    | 64 |
| Número de camadas    | 3 |
| BatchNorm            | Sim |
| Função de ativação   | ReLU |
| Dropout              | 0.5 |
| Learning rate        | 0.005 |
| Weight decay         | 5e-4 |
| Otimizador           | Adam |
| Loss                 | BCEWithLogitsLoss (com pos_weight dinâmico) |
| Estratégia temporal  | Incremental (t → t+1) |
| Épocas por timestep  | 20 |
| Grafo                | Não-direcionado + self-loops |


Tabela X – Hiperparâmetros do modelo GraphSAGE Temporal

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Hidden dimensions    | 64 |
| Número de camadas    | 2 |
| Função de ativação   | ReLU |
| Dropout              | 0.5 |
| Learning rate        | 0.005 |
| Weight decay         | 5e-4 |
| Otimizador           | Adam |
| Loss                 | BCEWithLogitsLoss (com pos_weight dinâmico) |
| Estratégia temporal  | Incremental (t → t+1) |
| Épocas por timestep  | 20 |
| Grafo                | Não-direcionado |



Tabela X – Hiperparâmetros do modelo GAT Temporal

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Hidden dimensions    | 32 |
| Heads (1ª camada)    | 4 |
| Heads (2ª camada)    | 1 |
| Função de ativação   | ELU |
| Dropout              | 0.5 |
| Learning rate        | 0.005 |
| Weight decay         | 5e-4 |
| Otimizador           | Adam |
| Loss                 | BCEWithLogitsLoss (com pos_weight dinâmico) |
| Estratégia temporal  | Incremental (t → t+1) |
| Épocas por timestep  | 20 |
| Grafo                | Não-direcionado |

Tabela X – Hiperparâmetros do modelo EvolveGCN Temporal

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Hidden dimensions    | 64 |
| Tipo de RNN          | GRU |
| Número de camadas RNN| 1 |
| Input                | Sequência temporal de features |
| Learning rate        | 0.005 |
| Weight decay         | 5e-4 |
| Otimizador           | Adam |
| Loss                 | BCEWithLogitsLoss (com pos_weight dinâmico) |
| Estratégia temporal  | Sequencial (estado oculto evolutivo) |
| Épocas por timestep  | 20 |



Tabela X – Hiperparâmetros do modelo TGN Temporal (Simplificado)

| Hiperparâmetro        | Valor |
|----------------------|-------|
| Dimensão da memória  | 64 |
| Tipo de atualização  | GRU |
| Memória              | Global (não por nó) |
| Message passing      | Não utilizado |
| Embedding temporal   | Implícito via estado |
| Learning rate        | 0.005 |
| Weight decay         | 5e-4 |
| Otimizador           | Adam |
| Loss                 | BCEWithLogitsLoss (com pos_weight dinâmico) |
| Estratégia temporal  | Incremental (t → t+1) |
| Épocas por timestep  | 20 |


“Os modelos foram treinados utilizando uma estratégia temporal incremental, na qual o grafo é construído com dados até o tempo t e utilizado para prever as transações no tempo t+1. O peso da classe positiva foi ajustado dinamicamente a cada iteração, a fim de lidar com o forte desbalanceamento do conjunto de dados.”