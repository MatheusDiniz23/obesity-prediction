"""
Script de Treinamento e Avaliação de Modelos - Modo Diagnóstico
--------------------------------------------------------------
Este script realiza o treinamento de múltiplos modelos de classificação
para apoio ao diagnóstico clínico, incluindo a variável 'Weight' no
pré-processamento.

Modelos: Regressão Logística, Random Forest e SVM.
Métrica Principal: F1-Score (Weighted).
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Importar o pré-processador do script de pré-processamento
from preprocessing import get_preprocessor

def load_data(path='../data/obesity.csv'):
    """Carrega o dataset e separa em features (X) e alvo (y)."""
    if not os.path.exists(path):
        # Tentar caminho alternativo se executado de src/
        path = 'data/obesity.csv' if os.path.exists('data/obesity.csv') else '../data/obesity.csv'
        
    df = pd.read_csv(path)
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    return X, y

def train_and_evaluate():
    """Executa o pipeline de treinamento e avaliação em modo diagnóstico."""
    
    # 1. Carregar dados
    X, y = load_data()
    
    # 2. Codificar a variável alvo (NObeyesdad)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # 3. Divisão Treino/Teste (Stratified para manter proporção das classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. Obter pré-processador em modo diagnóstico (Incluindo peso)
    preprocessor = get_preprocessor(include_weight=True)
    
    # 5. Definir Modelos (Torneio de Modelos)
    models = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42)
    }
    
    results = []
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None
    
    print(f"{'Modelo (Diagnóstico)':<25} | {'F1 (Mean)':<10} | {'Acc (Mean)':<10}")
    print("-" * 55)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Criar pipeline com pré-processamento e modelo
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Validação Cruzada
        cv_results = cross_validate(
            pipeline, X_train, y_train, 
            cv=skf, 
            scoring=['f1_weighted', 'accuracy'],
            return_train_score=False
        )
        
        mean_f1 = np.mean(cv_results['test_f1_weighted'])
        mean_acc = np.mean(cv_results['test_accuracy'])
        
        print(f"{name:<25} | {mean_f1:<10.4f} | {mean_acc:<10.4f}")
        
        # Seleção do melhor modelo baseado no F1-Score
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name
            best_pipeline = pipeline
            
    print("-" * 55)
    print(f"Melhor Modelo Diagnóstico: {best_model_name} (F1: {best_f1:.4f})")
    
    # 6. Avaliação Final no Conjunto de Teste
    print("\n--- Avaliação Final no Conjunto de Teste (Modo Diagnóstico) ---")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    
    print(f"\nModelo: {best_model_name}")
    print(f"Acurácia Final: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 7. Salvar Artefatos
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model_diagnostic.pkl'
    le_path = 'models/label_encoder.pkl'
    
    joblib.dump(best_pipeline, model_path)
    joblib.dump(le, le_path)
    
    print(f"\nArtefatos salvos em 'models/':")
    print(f"- Pipeline Diagnóstica: {model_path}")
    print(f"- Label Encoder: {le_path}")

if __name__ == "__main__":
    train_and_evaluate()
