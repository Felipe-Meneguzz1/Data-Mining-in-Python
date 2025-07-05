import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    print("🔎 Carregando dataset do UCI Repository...")
    wine = fetch_ucirepo(id=186)

    X = wine.data.features
    y_raw = wine.data.targets['quality']

    print("\n📈 Informações do dataset:")
    print(f"Registros: {X.shape[0]} | Atributos: {X.shape[1]}")
    print("\nPrimeiras linhas do dataset:")
    print(pd.concat([X, y_raw], axis=1).head())

    y = y_raw.apply(lambda x: 1 if x >= 7 else 0)

    print("\n📊 Distribuição dos vinhos:")
    print(y.value_counts().rename({0: "Ruins (<7)", 1: "Bons (>=7)"}))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Acurácia do modelo: {acc:.2%}")

    print("\n📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=["Ruim (<7)", "Bom (>=7)"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ruim", "Bom"], yticklabels=["Ruim", "Bom"])
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("matriz_confusao.png")
    plt.show()

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Importância das Variáveis")
    sns.barplot(x=importances[indices], y=X.columns[indices], hue=None, legend=False, palette="crest")
    plt.tight_layout()
    plt.savefig("importancia_atributos.png")
    plt.show()

    print("\n📦 Análises e gráficos salvos com sucesso!")

if __name__ == "__main__":
    main()
