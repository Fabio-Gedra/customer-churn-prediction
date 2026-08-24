# Previsão de Churn de Clientes

[🇺🇸 English Version](README.md)

## Visão Geral do Projeto

Este projeto analisa o cancelamento de clientes utilizando o dataset **IBM Telco Customer Churn**.

O principal objetivo é identificar características associadas ao churn e desenvolver um modelo de Machine Learning capaz de identificar clientes com maior probabilidade de deixar a empresa.

O projeto combina **Análise Exploratória de Dados (EDA)** com um modelo de **Regressão Logística**, buscando não apenas desempenho preditivo, mas também a interpretação dos principais padrões relacionados ao churn.

---

## Objetivos

Os principais objetivos deste projeto são:

- Explorar o dataset e identificar padrões relacionados ao churn.
- Analisar como características como tipo de contrato, tempo de permanência, serviço de internet, método de pagamento e cobranças mensais se relacionam com o cancelamento.
- Preparar variáveis categóricas e numéricas para Machine Learning.
- Tratar o desbalanceamento das classes utilizando SMOTE.
- Treinar um modelo de Regressão Logística como baseline interpretável de classificação.
- Avaliar o modelo utilizando diferentes métricas de classificação.
- Transformar os resultados da análise em insights que possam apoiar estratégias de retenção de clientes.

---

## Dataset

O projeto utiliza o dataset **IBM Telco Customer Churn**.

**Fonte original do dataset:** <https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data>

O dataset original contém:

- **7.043 clientes**
- **33 variáveis**
- Informações demográficas
- Informações da conta
- Informações contratuais
- Serviços de internet e telefonia
- Informações de pagamento
- Cobranças mensais e totais
- Informações relacionadas ao churn

O repositório já contém a **versão CSV utilizada neste projeto**, localizada em `data/Telco_customer_churn.csv`. Portanto, não é necessário baixar o arquivo original do Kaggle para executar o notebook.

Durante a etapa de limpeza, as colunas de cobrança foram convertidas para valores numéricos e registros com valores inválidos em `Total Charges` foram removidos, resultando em **7.032 observações** utilizadas na análise.

A variável alvo utilizada no modelo de Machine Learning é `Churn Value`, onde:

- `0` = Cliente não cancelou
- `1` = Cliente cancelou

---

## Análise Exploratória de Dados

A análise exploratória investigou a relação entre churn e diferentes características dos clientes.

### Distribuição de Churn

Aproximadamente:

- **73,42%** dos clientes permaneceram na empresa.
- **26,58%** dos clientes cancelaram.

Essa distribuição indica um desbalanceamento moderado entre as classes.

### Tipo de Contrato

O tipo de contrato apresentou uma das relações mais fortes com churn.

| Tipo de Contrato | Taxa de Churn |
| --- | ---: |
| Month-to-month | 42,71% |
| One year | 11,28% |
| Two year | 2,85% |

![Churn de Clientes por Tipo de Contrato](images/churn_by_contract.png)

Clientes com contratos **month-to-month** apresentaram uma taxa de churn consideravelmente maior, enquanto contratos de longo prazo estiveram associados a uma retenção significativamente maior.

### Serviço de Internet

As taxas de churn também apresentaram diferenças relevantes entre os tipos de serviço de internet.

| Serviço de Internet | Taxa de Churn |
| --- | ---: |
| Fiber optic | 41,89% |
| DSL | 19,00% |
| No internet service | 7,43% |

![Churn de Clientes por Serviço de Internet](images/churn_by_internet_service.png)

Clientes que utilizavam **Fiber Optic** apresentaram uma taxa de churn consideravelmente maior quando comparados aos clientes DSL e aos clientes sem serviço de internet.

### Método de Pagamento

Clientes que utilizavam **Electronic Check** apresentaram a maior taxa de churn entre os métodos de pagamento:

| Método de Pagamento | Taxa de Churn |
| --- | ---: |
| Electronic check | 45,29% |
| Mailed check | 19,20% |
| Bank transfer (automatic) | 16,73% |
| Credit card (automatic) | 15,25% |

### Tempo de Permanência

O tempo de permanência do cliente apresentou uma relação importante com churn.

Clientes que cancelaram apresentaram uma mediana de aproximadamente **10 meses de permanência**, enquanto clientes que permaneceram apresentaram uma mediana de aproximadamente **38 meses**.

Esse resultado sugere que clientes nos estágios iniciais do relacionamento com a empresa apresentam maior vulnerabilidade ao churn.

### Cobranças Mensais

Clientes que cancelaram apresentaram, em geral, cobranças mensais mais elevadas.

A mediana de `Monthly Charges` foi de aproximadamente:

- **79,65** para clientes que cancelaram
- **64,45** para clientes que permaneceram

### Análise de Correlação

A análise de correlação apresentou as seguintes relações:

- `Tenure Months` e `Churn Value`: **-0,35**
- `Monthly Charges` e `Churn Value`: **0,19**
- `Total Charges` e `Churn Value`: **-0,20**

Também foram observadas correlações fortes entre:

- `Tenure Months` e `Total Charges`: **0,83**
- `Monthly Charges` e `Total Charges`: **0,65**

Essas relações foram consideradas posteriormente durante a interpretação do modelo de Machine Learning.

---

## Pré-processamento dos Dados

Antes do treinamento do modelo, foram realizadas as seguintes etapas:

1. Conversão de `Monthly Charges` e `Total Charges` para valores numéricos.
2. Remoção de observações com valores inválidos em `Total Charges`.
3. Remoção de identificadores, variáveis geográficas e informações que poderiam causar vazamento de dados.
4. Separação entre as variáveis preditoras e a variável alvo `Churn Value`.
5. Codificação das variáveis categóricas utilizando one-hot encoding com `pandas.get_dummies()`.
6. Divisão do dataset em:
   - **80% para treinamento**
   - **20% para teste**
7. Utilização de amostragem estratificada para preservar a proporção de churn entre os conjuntos de treino e teste.
8. Padronização das seguintes variáveis:
   - `Tenure Months`
   - `Monthly Charges`
   - `Total Charges`
9. Aplicação de **SMOTE** somente no conjunto de treinamento para tratar o desbalanceamento das classes.

Após o processo de codificação, o modelo utilizou **30 features**.

---

## Modelo de Machine Learning

### Regressão Logística

A Regressão Logística foi escolhida como modelo baseline por combinar capacidade preditiva com boa interpretabilidade.

O modelo foi treinado utilizando o conjunto de treinamento balanceado com SMOTE e avaliado utilizando o conjunto de teste original, sem aplicação de oversampling.

---

## Desempenho do Modelo

O modelo de Regressão Logística apresentou os seguintes resultados:

| Métrica | Resultado |
| --- | ---: |
| Accuracy | **75,76%** |
| Precision | **53,13%** |
| Recall | **74,87%** |
| F1-Score | **62,15%** |
| ROC-AUC | **83,49%** |

Como churn é um problema de classificação desbalanceada, métricas como **Recall, F1-Score e ROC-AUC** são especialmente importantes para avaliar o desempenho do modelo.

O Recall de aproximadamente **74,9%** indica que o modelo conseguiu identificar quase **três em cada quatro clientes que realmente cancelaram**.

O ROC-AUC de aproximadamente **0,835** também demonstra uma boa capacidade de diferenciação entre clientes que cancelam e clientes que permanecem.

---

### Curva ROC

![Curva ROC da Regressão Logística](images/roc_curve.png)

A curva ROC demonstra a capacidade do modelo de distinguir entre clientes que cancelam e clientes que permanecem ao longo de diferentes thresholds de classificação.

O modelo atingiu um **ROC-AUC de aproximadamente 0,835**, indicando um desempenho de discriminação consideravelmente superior a uma classificação aleatória.

---

## Matriz de Confusão

![Matriz de Confusão da Regressão Logística](images/confusion_matrix.png)

No conjunto de teste, o modelo apresentou:

| | Previsto: Sem Churn | Previsto: Churn |
| --- | ---: | ---: |
| **Real: Sem Churn** | 786 | 247 |
| **Real: Churn** | 94 | 280 |

O modelo identificou corretamente:

- **786 clientes que não cancelaram**
- **280 clientes que cancelaram**

O modelo deixou de identificar **94 clientes que realmente cancelaram**, enquanto **247 clientes que permaneceram** foram classificados como possíveis churners.

Do ponto de vista de retenção, o Recall relativamente alto pode ser útil, pois permite identificar uma parcela significativa dos clientes potencialmente em risco e direcionar ações preventivas para esse grupo.

---

## Coeficientes da Regressão Logística

Os coeficientes da Regressão Logística foram analisados para compreender melhor as relações aprendidas pelo modelo.

Alguns resultados foram consistentes com os padrões encontrados durante a análise exploratória:

- **Maior tempo de permanência** apresentou associação negativa com churn.
- **Contratos de dois anos** apresentaram associação negativa com churn.
- **Serviço de internet Fiber Optic** apresentou forte associação positiva com churn.

Algumas variáveis relacionadas aos serviços contratados possuem correlação entre si e também com os valores cobrados. Devido a essa multicolinearidade, a magnitude e a direção de determinados coeficientes devem ser interpretadas com cautela.

Por exemplo, apesar de cobranças mensais mais elevadas estarem associadas a maior churn durante a análise exploratória, o coeficiente de `Monthly Charges` tornou-se negativo na Regressão Logística quando as demais características e serviços dos clientes foram mantidos constantes.

Portanto, os coeficientes devem ser interpretados como relações dentro do modelo, e não como efeitos causais diretos.

---

## Estrutura do Projeto

```text
customer-churn-prediction/
│
├── data/
│   └── Telco_customer_churn.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── images/
│   ├── churn_by_contract.png
│   ├── churn_by_internet_service.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
├── README.pt-BR.md
└── requirements.txt
```
---
## Tecnologias

O projeto foi desenvolvido utilizando:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- Jupyter Notebook

---

## Como Executar o Projeto

### 1. Instalar as bibliotecas necessárias

```bash
pip install -r requirements.txt
```

### 2. Dataset

O arquivo CSV utilizado pelo notebook já está incluído no repositório em:

```text
data/Telco_customer_churn.csv
```

Não é necessário realizar nenhum download adicional do dataset.

### 3. Abrir o notebook

```bash
jupyter notebook
```

Em seguida, abra:

```text
notebooks/churn_analysis.ipynb
```

e execute as células em ordem.

---

## Conclusão

A análise identificou diferentes características associadas ao churn de clientes.

Clientes com **menor tempo de permanência**, **contratos month-to-month**, **serviço de internet Fiber Optic** e determinados métodos de pagamento apresentaram taxas mais elevadas de churn durante a análise exploratória.

O modelo de Regressão Logística alcançou um **ROC-AUC de aproximadamente 83,5%** e identificou aproximadamente **74,9% dos clientes que efetivamente cancelaram**, fornecendo um baseline interpretável e útil para previsão de churn.

O projeto demonstra como Análise Exploratória de Dados e Machine Learning podem ser combinados para identificar padrões de comportamento dos clientes e apoiar potenciais estratégias de retenção.

Possíveis melhorias futuras incluem:

- Otimização do threshold de classificação
- Engenharia adicional de features
- Comparação com outros algoritmos de classificação
- Otimização de hiperparâmetros