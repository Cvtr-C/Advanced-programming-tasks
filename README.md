# Classificação de Diabetes com KNN e Random Forest

Projeto desenvolvido para a disciplina de Programação Avançada com o objetivo de aplicar técnicas de análise exploratória e Machine Learning em um conjunto de dados relacionado ao diabetes.

Foram utilizados os algoritmos **K-Nearest Neighbors (KNN)** e **Random Forest**, comparando seus desempenhos por meio de diferentes métricas de classificação.

## Dataset

O conjunto de dados utilizado foi obtido no Kaggle:

[Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/diabetes)

O dataset contém informações clínicas e físicas dos indivíduos, como:

- colesterol;
- glicose estável;
- HDL;
- idade;
- peso;
- altura;
- pressão arterial;
- circunferência da cintura;
- circunferência do quadril;
- hemoglobina glicada.

Como o conjunto não possuía uma variável-alvo explícita para classificação, foi criada a variável `index` utilizando a hemoglobina glicada (`glyhb`).

Foi utilizado o valor de **6,5%** como ponto de corte:

- `0`: glyhb < 6,5%;
- `1`: glyhb ≥ 6,5%.

## Análise Exploratória

Durante a análise exploratória foram realizadas etapas como:

- verificação de registros duplicados;
- identificação de valores ausentes;
- análise estatística das variáveis;
- análise da distribuição das classes;
- histogramas;
- gráficos de dispersão;
- matriz de correlação;
- levantamento e análise de hipóteses sobre os dados.

Também foi identificado um desbalanceamento entre as classes, com aproximadamente:

- **83,33%** dos registros na classe 0;
- **16,67%** dos registros na classe 1.

## Pré-processamento

Antes do treinamento dos modelos foram realizadas as seguintes etapas:

- divisão dos dados em **80% para treinamento e 20% para teste**;
- manutenção da proporção das classes com `stratify`;
- tratamento dos valores ausentes;
- preenchimento pela mediana nas variáveis numéricas;
- preenchimento pela moda nas variáveis categóricas;
- codificação das variáveis categóricas com `OneHotEncoder`;
- padronização das características utilizadas pelo KNN com `StandardScaler`;
- balanceamento do conjunto de treinamento utilizando `RandomOverSampler`.

## Modelos

### K-Nearest Neighbors

O KNN foi configurado com:

```python
KNeighborsClassifier(
    n_neighbors=21,
    weights="uniform",
    metric="minkowski",
    p=2
)
```

Resultado obtido:

**Acurácia: 87,18%**

Para a classe 1:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |      0,60 |
| Recall    |      0,69 |
| F1-score  |      0,64 |

### Random Forest

O Random Forest foi configurado com:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    min_samples_split=6,
    class_weight="balanced_subsample",
    random_state=42
)
```

Resultado obtido:

**Acurácia: 94,87%**

Para a classe 1:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |      0,85 |
| Recall    |      0,85 |
| F1-score  |      0,85 |

## Comparação dos Modelos

| Modelo        | Acurácia |
| ------------- | -------: |
| KNN           |   87,18% |
| Random Forest |   94,87% |

Os dois modelos alcançaram acurácia superior a 85%, porém o **Random Forest apresentou o melhor desempenho geral**, além de resultados superiores na identificação da classe minoritária.

Por esse motivo, o Random Forest foi escolhido como modelo final.

## Importância das Características

Entre as características consideradas mais importantes pelo Random Forest estão:

| Característica | Importância aproximada |
| -------------- | ---------------------: |
| `stab.glu`     |                  33,5% |
| `age`          |                  15,6% |
| `bp.1s`        |                   7,7% |
| `chol`         |                   5,3% |
| `waist`        |                   5,3% |
| `ratio`        |                   4,9% |

## Modelo Salvo

O melhor modelo foi salvo utilizando a biblioteca `joblib`:

```python
joblib.dump(clf, "modelo_rf.joblib")
```

O arquivo `modelo_rf.joblib` está disponível neste repositório.

## Arquivos do Repositório

```text
.
├── KNN_Random_Forest.ipynb
├── modelo_rf.joblib
└── README.md
```

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Jupyter Notebook / Google Colab

## Execução

O projeto pode ser executado utilizando Jupyter Notebook ou Google Colab.

As principais dependências podem ser instaladas com:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib kagglehub
```

Depois, basta abrir e executar o arquivo:

```text
KNN_Random_Forest.ipynb
```

## Conclusão

A análise mostrou que os dois modelos conseguiram atingir desempenho superior ao mínimo estabelecido para o trabalho.

O KNN alcançou **87,18% de acurácia**, enquanto o Random Forest atingiu **94,87%**, apresentando também melhores valores de precisão, recall e F1-score.

Com base nesses resultados, o **Random Forest foi selecionado como o melhor modelo** e salvo para utilização posterior.

## 👨‍💻 Autor

Desenvolvido por **Carlos Vitor Taleires Rodrigues**

Trabalho de Machine Learning desenvolvido para a disciplina de Programação Avançada, com foco na classificação de dados relacionados ao diabetes utilizando os algoritmos KNN e Random Forest.
