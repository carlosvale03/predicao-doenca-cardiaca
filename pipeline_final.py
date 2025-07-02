import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix 
import warnings

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


warnings.filterwarnings('ignore', category=UserWarning)

print("--- Iniciando Múltiplas Execuções para Análise Robusta (Métricas Detalhadas) ---")

models = {
    "KNN": KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform'),
    "SVM": SVC(C=100, gamma='scale', kernel='linear', random_state=42),
    "Random Forest": RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=2, n_estimators=300, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', alpha=0.01, learning_rate_init=0.001, max_iter=500, random_state=42)
}

results = {name: [] for name in models.keys()}
n_runs = 30

for i in range(n_runs):
    print(f"--- Execução {i+1}/{n_runs} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=i, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results[name].append({
            'Acurácia': report['accuracy'],
            'Precisão (0)': report['0']['precision'], 'Recall (0)': report['0']['recall'], 'F1-Score (0)': report['0']['f1-score'],
            'Precisão (1)': report['1']['precision'], 'Recall (1)': report['1']['recall'], 'F1-Score (1)': report['1']['f1-score'],
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })

# --- 3. CÁLCULO DAS ESTATÍSTICAS E TABELAS FINAIS ---
summary_results = []
for name, run_metrics in results.items():
    run_df = pd.DataFrame(run_metrics)
    mean_scores = run_df.mean()
    std_scores = run_df.std()
    
    summary_results.append({
        "Algoritmo": name,
        "Acurácia": f"{mean_scores['Acurácia']:.2%} ± {std_scores['Acurácia']:.2%}",
        "Precisão (0)": f"{mean_scores['Precisão (0)']:.2%} ± {std_scores['Precisão (0)']:.2%}",
        "Recall (0)": f"{mean_scores['Recall (0)']:.2%} ± {std_scores['Recall (0)']:.2%}",
        "F1-Score (0)": f"{mean_scores['F1-Score (0)']:.2%} ± {std_scores['F1-Score (0)']:.2%}",
        "Precisão (1)": f"{mean_scores['Precisão (1)']:.2%} ± {std_scores['Precisão (1)']:.2%}",
        "Recall (1)": f"{mean_scores['Recall (1)']:.2%} ± {std_scores['Recall (1)']:.2%}",
        "F1-Score (1)": f"{mean_scores['F1-Score (1)']:.2%} ± {std_scores['F1-Score (1)']:.2%}"
    })
final_results_df = pd.DataFrame(summary_results)
final_results_df['acc_mean_sorter'] = [np.mean([m['Acurácia'] for m in results[name]]) for name in final_results_df['Algoritmo']]
final_results_df = final_results_df.sort_values(by='acc_mean_sorter', ascending=False).drop(columns='acc_mean_sorter').reset_index(drop=True)

print("\n\n--- Tabela de Resultados Finais (Média de 30 Execuções) ---")
print(final_results_df.to_string())


avg_cm_results = []
for name, run_metrics in results.items():
    run_df = pd.DataFrame(run_metrics)
    mean_cm = run_df[['TN', 'FP', 'FN', 'TP']].mean()
    
    avg_cm_results.append({
        "Algoritmo": name,
        "Verdadeiro Negativo (Média)": f"{mean_cm['TN']:.2f}",
        "Falso Positivo (Média)": f"{mean_cm['FP']:.2f}",
        "Falso Negativo (Média)": f"{mean_cm['FN']:.2f}",
        "Verdadeiro Positivo (Média)": f"{mean_cm['TP']:.2f}"
    })
    
avg_cm_df = pd.DataFrame(avg_cm_results)
avg_cm_df = pd.merge(final_results_df[['Algoritmo']], avg_cm_df, on='Algoritmo', how='left')

print("\n\n--- Tabela de Matriz de Confusão Média (Valores Médios de 30 Execuções) ---")
print(avg_cm_df.to_string())