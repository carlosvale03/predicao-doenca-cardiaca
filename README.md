# Análise Preditiva de Doença Cardíaca com Machine Learning

Este repositório contém o código e a análise desenvolvidos para o projeto final da disciplina de Sistemas Inteligentes. O objetivo principal do projeto é realizar uma análise comparativa de diferentes modelos de aprendizado de máquina para a predição do risco de doença arterial coronariana.

## 📋 Sobre o Projeto

O estudo segue um pipeline completo de Machine Learning, que inclui:
1.  **Análise Exploratória de Dados (EDA)** para entender as características e correlações do dataset.
2.  **Seleção de Atributos** rigorosa com a técnica de RFECV para determinar o conjunto ideal de preditores.
3.  **Otimização de Hiperparâmetros** para quatro algoritmos de classificação distintos (KNN, SVM, Random Forest e MLP) usando `GridSearchCV`.
4.  **Avaliação Robusta** através da execução de 30 rodadas independentes para garantir a validade estatística dos resultados, reportando a média e o desvio padrão das principais métricas.

## 📊 Dataset

O estudo utiliza o **"Heart Attack Analysis & Prediction Dataset"**, uma versão pré-processada e publicamente disponível, derivada do renomado dataset "Heart Disease" do Repositório da UC Irvine.

- **Fonte dos Dados:** [Link para o dataset no GitHub](https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv)
- **Artigo de Referência Principal:** Janaranjani, N., et al. (2022). "Heart attack prediction using machine learning". *ICIRCA 2022*.

## 🚀 Resultados Finais

Após um processo completo de otimização e avaliação robusta sobre 30 execuções, a performance comparativa dos modelos foi consolidada. O SVM Otimizado alcançou a maior acurácia média, embora todos os modelos tenham apresentado um desempenho competitivo e muito similar, indicando uma convergência de performance.

| Algoritmo | Acurácia | Precisão (Classe 1) | Recall (Classe 1) | F1-Score (Classe 1) |
| :--- | :--- | :--- | :--- | :--- |
| SVM (Otimizado) | 84.37% ± 4.57% | 82.37% ± 4.42% | 90.71% ± 5.62% | 86.25% ± 4.07% |
| Random Forest (Otimizado) | 84.21% ± 4.40% | 83.04% ± 4.54% | 89.29% ± 5.83% | 85.93% ± 3.99% |
| KNN (Otimizado) | 83.66% ± 4.00% | 81.45% ± 4.34% | 90.71% ± 4.84% | 85.73% ± 3.49% |
| MLP (Otimizado) | 83.44% ± 4.20% | 83.86% ± 4.73% | 86.26% ± 5.83% | 84.90% ± 3.93% |

## 🛠️ Como Executar o Projeto

Siga os passos abaixo para replicar os resultados.

### 1. Pré-requisitos
- Python 3.9+
- Git

### 2. Instalação

Primeiro, clone o repositório para a sua máquina local:
```bash
git clone [https://github.com/carlosvale03/predicao-doenca-cardiaca.git](https://github.com/carlosvale03/predicao-doenca-cardiaca.git)
cd predicao-doenca-cardiaca
```

É altamente recomendado criar um ambiente virtual:
```bash
python -m venv venv
# No Windows:
# venv\Scripts\activate
# No macOS/Linux:
# source venv/bin/activate
```

Instale as dependências necessárias a partir do arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Execução dos Scripts

O projeto está dividido em dois arquivos principais:

- **`otimizacao_modelos.py`**: Este script realiza a busca exaustiva (`GridSearchCV`) para encontrar os melhores hiperparâmetros para cada modelo. Ele documenta o processo de otimização.
    ```bash
    python otimizacao_modelos.py
    ```

- **`pipeline_final.py`**: Este é o script principal que executa as 30 rodadas do experimento usando os modelos já otimizados. Ele gera as tabelas de resultados finais que foram usadas no artigo.
    ```bash
    python pipeline_final.py
    ```

## 📂 Estrutura do Repositório
```
.
├── otimizacao_modelos.py     # Script para a busca de hiperparâmetros
├── pipeline_final.py         # Script para a avaliação final e geração de resultados
└── README.md                 # Este arquivo
```

## ✍️ Autores
- Carlos Henrique
- Eduardo de Sousa
- Rafael Barbosa