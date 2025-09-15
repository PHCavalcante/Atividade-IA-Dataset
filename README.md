# Global Crocodile Species — Classificação do Status de Conservação 🐊

![License](https://img.shields.io/badge/license-MIT-green.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)

> Atividade avaliativa da máteria de Inteligência Artificial do curso de Sistemas de Informação no Centro Universitário do Rio São Francisco - Unirios. Este projeto trata da **classificação** aplicada ao dataset *Global Crocodile Species* (Kaggle). O objetivo é prever o **status de conservação** a partir de características observacionais e ambientais.

---

## Índice
- [Sobre o dataset](#sobre-o-dataset)
- [Justificativa de escolha](#justificativa-de-escolha)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Preview dos dados](#preview-dos-dados)
- [Resumo do pipeline](#resumo-do-pipeline)
- [Resultados principais](#resultados-principais-artifact-artifactsmetricscsv)
- [Resultado gráfico](#resultado-gráfico)
- [Questões conclusivas](#questões-conclusivas)
- [Créditos e referência](#créditos-e-referência)
- [Licença](#licença)

## Sobre o dataset
- **Fonte:** [Kaggle — Global Crocodile Species Dataset](https://www.kaggle.com/datasets/zadafiyabhrami/global-crocodile-species-dataset)  
- **Target escolhido:** `Conservation Status` (status de conservação; problema de classificação multiclasse)  
- **Tamanho aproximado:** ~1000 observações (linhas)  
- **Principais colunas (exemplos):**
  - `Common Name`, `Scientific Name`, `Family`, `Genus`
  - `Observed Length (m)`, `Observed Weight (kg)`
  - `Age Class`, `Sex`, `Date of Observation`
  - `Country/Region`, `Habitat Type`
  - `Conservation Status`
- **Observação importante:** Algumas colunas (ex.: `Scientific Name`, `Genus`, `Family`) revelam diretamente a espécie, que por sua vez determina o status, por isso foram removidas do conjunto de features para evitar vazamento.

## Justificativa de escolha
A escolha do presente dataset foi dada por alguns motivos, sendo eles, o tema que me chamou bastante atenção por ser diferenciado, a diversificação de targets, o tamanho, e para tentar fugir do padrão esperado (Wine, Breast Cancer, etc).

## Estrutura do repositório
```
ia-atividade/
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ data/
│  └─ crocodile_dataset.csv
├─ src/
│  ├─ data_load.py
│  ├─ preprocessing.py
│  └─ train_eval.py
├─ artifacts/   # outputs gerados (metrics, imagens, modelos)
└─ notebooks/
   └─ atividade_IA.ipynb
```

## Preview dos dados
```text
Shape: (aprox. 1001, N)
Colunas: ['Observation ID','Common Name','Scientific Name','Family','Genus',
          'Observed Length (m)','Observed Weight (kg)','Age Class','Sex',
          'Date of Observation','Country/Region','Habitat Type','Conservation Status',
          'Observer Name','Notes']
```

## Resumo do pipeline
1. **Carregamento** do CSV (arquivo local `data/crocodile_dataset.csv`).
2. **Exploração inicial**: checagem de nulos, duplicados e distribuição das classes.
3. **Remoção de colunas que causam vazamento**: `Scientific Name`, `Common Name`, `Genus`, `Family`.
4. **Pré-processamento:**
   - Numéricas: imputação (mediana) + `StandardScaler`.
   - Categóricas: imputação (mais frequente) + `OneHotEncoder`.
5. **Divisão treino/teste:** 75% / 25% (com `stratify` para preservar proporção das classes).
6. **Modelos treinados (baseline):**
   - Decision Tree
   - KNN (k = 5)
   - Logistic Regression
7. **Avaliação:** Accuracy, Precision, Recall, F1 (média ponderada para multiclass), matriz de confusão.

---

## Resultados principais (artifact: `artifacts/metrics.csv`)
| Modelo               | Acurácia | F1     | Precision | Recall |
|---------------------:|:--------:|:------:|:---------:|:------:|
| Decision Tree        | 0.956    | 0.955924314938404    | 0.956383214935774       | 0.956    |
| KNN (k=5)            | 0.820    | 0.8133355605077641    | 0.8325836544292579       | 0.82    |
| Logistic Regression  | 0.996    | 0.996,0.9959862663087583    | 0.9960412371134021       | 0.996    |


---

## Resultado gráfico

<img src="/results/accuracy_comparison.png" alt="Comparação de Acurácias" width="700px" style="display: block; margin: 0 auto" />

## Questões conclusivas

- Qual modelo apresentou melhor desempenho no dataset escolhido?
No presente experimento com o dataset escolhido (Global Crocodile Species Dataset), o modelo que apresentou melhor desempenho foi o **Logistic Regression**, que apresentou um score bastante alto, com cerca de **0.996 pontos**, seguido pelo **Decision Tree** com **0.956** e por fim o **KNN** com **0.820**.

- O resultado faz sentido? (justificar com base nos dados).
**Sim**, pois o dataset apresenta variáveis com boa separabilidade linear, favorecendo a regressão logística. O modelo **KNN** foi prejudicado devido ao grande número de features categóricas.

- O que poderia ser feito para melhorar os modelos (ex: normalização, mais features, tuning de
hiperparâmetros)?
Os modelos podem ser melhorados de algumas formas: Primeiramente ajustando pontos críticos em cada algoritmo, por exemplo, a profundidade da árvore, o número de vizinhos no KNN, e os parâmetros da regressão logística. Também seria interessante repetir os testes várias vezes, dividindo os dados de formas diferentes para ter certeza de que o resultado não foi sorte de uma divisão. Por fim, poderiamos reduzir ou escolher mais sabiamente as colunas mais importantes, para evitar que os modelos fiquem confusos com informações repetidas.

## Créditos e referência
- Dataset: Zadafiyabhrami — *Global Crocodile Species Dataset*, Kaggle.  
  https://www.kaggle.com/datasets/zadafiyabhrami/global-crocodile-species-dataset

## Licença
Este projeto esta licenciado sobre a licença **MIT**. Para mais informações visite a [Licença]("/LICENSE").
