import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPClassifier


# URL para a versão "raw" do arquivo CSV no GitHub
raw_url = 'https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv'

print(f"--- Carregando dataset de: {raw_url} ---")

try:
    df = pd.read_csv(raw_url)
    print("Dataset carregado com sucesso!")

except Exception as e:
    print(f"\nOcorreu um erro ao carregar o dataset: {e}")
    print("Verifique se o link está correto e se há conexão com a internet.")
    

print("--- Iniciando Preparação Final dos Dados ---")

X = df.drop('target', axis=1)
y = df['target']
print(f"Features (X) e alvo (y) separados. Formato de X: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nDados padronizados com sucesso.")

print("--- Iniciando Análise de Importância de Features ---")

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_scaled, y)
print("Modelo Random Forest treinado para análise de importância.")

importances = rf_selector.feature_importances_
feature_names = X.columns 

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("--- Selecionando o subconjunto final de features ---")

N_FEATURES_TO_KEEP = 13

top_features = feature_importance_df.head(N_FEATURES_TO_KEEP)['Feature'].tolist()

print(f"\nAs {N_FEATURES_TO_KEEP} features selecionadas são:")
print(top_features)

X_final = X_scaled_df[top_features]

# Treinamento

print("--- Iniciando Etapa de Treinamento e Avaliação com Random Forest ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

print("\nIniciando a busca pelos melhores parâmetros... (Isso pode levar alguns minutos)")
grid_search.fit(X_train, y_train)

# 5. Apresentação dos Melhores Resultados
print("\n--- Resultados da Otimização ---")
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
print(f"Melhor acurácia durante a validação cruzada: {grid_search.best_score_:.4f}")

print("--- Iniciando Otimização de Hiperparâmetros para o SVM ---")

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

grid_search_svm = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid_svm,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

print("\nIniciando a busca pelos melhores parâmetros para o SVM...")
grid_search_svm.fit(X_train, y_train)

print("\n--- Resultados da Otimização do SVM ---")
print(f"Melhores parâmetros encontrados: {grid_search_svm.best_params_}")
print(f"Melhor acurácia durante a validação cruzada: {grid_search_svm.best_score_:.4f}")

print("--- Iniciando Otimização de Hiperparâmetros para o KNN ---")

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search_knn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_knn,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

print("\nIniciando a busca pelos melhores parâmetros para o KNN...")
grid_search_knn.fit(X_train, y_train)

print("\n--- Resultados da Otimização do KNN ---")
print(f"Melhores parâmetros encontrados: {grid_search_knn.best_params_}")
print(f"Melhor acurácia durante a validação cruzada: {grid_search_knn.best_score_:.4f}")


print("--- Iniciando Otimização de Hiperparâmetros para o MLP ---")

param_grid_mlp = {
    'hidden_layer_sizes': [(32, 16), (64, 32), (100,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid_search_mlp = GridSearchCV(
    estimator=MLPClassifier(max_iter=500, random_state=42),
    param_grid=param_grid_mlp,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

print("\nIniciando a busca pelos melhores parâmetros para o MLP...")
grid_search_mlp.fit(X_train, y_train)

print("\n--- Resultados da Otimização do MLP ---")
print(f"Melhores parâmetros encontrados: {grid_search_mlp.best_params_}")
print(f"Melhor acurácia durante a validação cruzada: {grid_search_mlp.best_score_:.4f}")





print("--- Consolidando os Resultados Finais da Otimização ---")

final_data = {
    'Algoritmo': [
        'Random Forest', 
        'Support Vector Machine (SVM)', 
        'K-Nearest Neighbors (KNN)', 
        'Rede Neural (MLP)'
    ],
    'Melhores Parâmetros': [
        str(grid_search.best_params_),
        str(grid_search_svm.best_params_),
        str(grid_search_knn.best_params_),
        str(grid_search_mlp.best_params_)
    ],
    'Acurácia Média (Validação Cruzada)': [
        grid_search.best_score_,
        grid_search_svm.best_score_,
        grid_search_knn.best_score_,
        grid_search_mlp.best_score_
    ]
}

final_summary_df = pd.DataFrame(final_data)

final_summary_df['Acurácia Média (Validação Cruzada)'] = final_summary_df['Acurácia Média (Validação Cruzada)'].map('{:.2%}'.format)

final_summary_df = final_summary_df.sort_values(by='Acurácia Média (Validação Cruzada)', ascending=False).reset_index(drop=True)

print("\n--- Tabela Comparativa Final (Modelos Otimizados) ---")
print(final_summary_df.to_string())