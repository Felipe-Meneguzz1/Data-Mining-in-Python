# 🍷 Wine Quality Classifier

Este projeto tem como objetivo aplicar uma técnica de **mineração de dados** usando Python para classificar a qualidade de vinhos tintos com base em características físico-químicas.

## 📂 Dataset

- **Nome:** Wine Quality
- **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Registros:** 6.497
- **Atributos:** 11

---

## ⚙️ Técnicas Aplicadas

- Classificação Binária: **Bom (>=7)** ou **Ruim (<7)**
- Algoritmo: `RandomForestClassifier` (scikit-learn)
- Divisão treino/teste: 80% / 20%

---

## 📊 Resultados

- **Acurácia:** ~88%
- **Matriz de Confusão:** salva como `matriz_confusao.png`
- **Importância dos Atributos:** salva como `importancia_atributos.png`

## 🚀 Como Executar

1. Clone o repositório:
```bash
  git clone https://github.com/seu-usuario/wine-quality-classifier.git
  cd wine-quality-classifier
```
2. Baixe as dependencias
```bash
    pip install -r requirements.txt
```
3. Rode o projeto
```bash
  python wine_analysis.py
```