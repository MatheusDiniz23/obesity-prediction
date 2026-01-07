"""
Script de Pré-processamento e Feature Engineering
-------------------------------------------------
Este script implementa a pipeline de transformação de dados para o projeto
de predição de obesidade, utilizando scikit-learn Pipelines e ColumnTransformer.

As definições seguem o dicionário de dados (docs/data_dictionary.pdf) e a
estratégia de prevenção de Data Leakage.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

def get_preprocessor(include_weight=False):
    """
    Cria e retorna o ColumnTransformer configurado para o pré-processamento.
    
    Args:
        include_weight (bool): Se True, inclui a variável 'Weight' no processamento.
                              Padrão é False para evitar Data Leakage em modelos preventivos.
    
    Returns:
        ColumnTransformer: Objeto configurado com as transformações por tipo de variável.
    """
    
    # 1. Definição das colunas por categoria (Baseado no Dicionário de Dados)
    
    # Numéricas Contínuas
    num_features = ['Age', 'Height']
    if include_weight:
        num_features.append('Weight')
        
    # Categóricas Nominais (Sem ordem inerente)
    nom_features = [
        'Gender', 
        'family_history_with_overweight', 
        'FAVC', 
        'SMOKE', 
        'SCC', 
        'MTRANS'
    ]
    
    # Categóricas Ordinais (Com hierarquia clara definida por variável)
    ord_features = ['CAEC', 'CALC']
    caec_categories = ['no', 'Sometimes', 'Frequently', 'Always']
    calc_categories = ['no', 'Sometimes', 'Frequently', 'Always']
    
    # Numéricas de Escala (Devem ser arredondadas conforme dicionário)
    # FCVC (1-3), NCP (1-4), CH2O (1-3), FAF (0-3), TUE (0-2)
    scale_features = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # 2. Funções Customizadas
    # Arredondamento para mitigar ruído sintético ou inconsistências em dados discretos
    rounder = FunctionTransformer(np.round)
    
    # 3. Construção dos Pipelines por tipo de dado
    
    # Pipeline para Numéricas: Imputação pela mediana + Padronização
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para Nominais: Imputação pelo mais frequente + One-Hot Encoding
    nom_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Pipeline para Ordinais: Imputação pelo mais frequente + Ordinal Encoding
    # Categorias definidas explicitamente para cada variável ordinal
    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[caec_categories, calc_categories]))
    ])
    
    # Pipeline para Escalas: Arredondamento + Padronização
    scale_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('round', rounder),
        ('scaler', StandardScaler())
    ])
    
    # 4. ColumnTransformer Final
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('nom', nom_pipeline, nom_features),
            ('ord', ord_pipeline, ord_features),
            ('scale', scale_pipeline, scale_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def preprocess_data(df, include_weight=False, fit=False, preprocessor=None):
    """
    Aplica o pré-processamento em um DataFrame.
    
    Args:
        df (pd.DataFrame): Dados de entrada.
        include_weight (bool): Se deve incluir o peso.
        fit (bool): Se deve ajustar o preprocessor aos dados.
        preprocessor (ColumnTransformer): Objeto preprocessor existente (opcional).
        
    Returns:
        tuple: (dados_transformados, preprocessor)
    """
    if preprocessor is None:
        preprocessor = get_preprocessor(include_weight=include_weight)
        
    if fit:
        transformed_data = preprocessor.fit_transform(df)
    else:
        transformed_data = preprocessor.transform(df)
        
    return transformed_data, preprocessor

if __name__ == "__main__":
    # Exemplo de uso rápido para validação
    print("Script de pré-processamento carregado.")
    print("As variáveis ordinais (CAEC, CALC) possuem hierarquias definidas explicitamente.")
    print("As variáveis de escala (FCVC, NCP, etc.) são arredondadas para mitigar ruído em dados discretos.")
