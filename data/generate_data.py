import pandas as pd
import numpy as np

def generate_obesity_data(n_samples=2111):
    np.random.seed(42)
    
    genders = ['Male', 'Female']
    family_history = ['yes', 'no']
    favc = ['yes', 'no']
    caec = ['Sometimes', 'Frequently', 'Always', 'no']
    smoke = ['yes', 'no']
    scc = ['yes', 'no']
    calc = ['no', 'Sometimes', 'Frequently', 'Always']
    mtrans = ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike']
    obesity_levels = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    data = {
        'Gender': np.random.choice(genders, n_samples),
        'Age': np.random.uniform(14, 61, n_samples),
        'Height': np.random.uniform(1.45, 1.98, n_samples),
        'Weight': np.random.uniform(39, 173, n_samples),
        'family_history_with_overweight': np.random.choice(family_history, n_samples),
        'FAVC': np.random.choice(favc, n_samples),
        'FCVC': np.random.uniform(1, 3, n_samples),
        'NCP': np.random.uniform(1, 4, n_samples),
        'CAEC': np.random.choice(caec, n_samples),
        'SMOKE': np.random.choice(smoke, n_samples),
        'CH2O': np.random.uniform(1, 3, n_samples),
        'SCC': np.random.choice(scc, n_samples),
        'FAF': np.random.uniform(0, 3, n_samples),
        'TUE': np.random.uniform(0, 2, n_samples),
        'CALC': np.random.choice(calc, n_samples),
        'MTRANS': np.random.choice(mtrans, n_samples),
        'NObeyesdad': np.random.choice(obesity_levels, n_samples)
    }

    df = pd.DataFrame(data)
    
    # Ajustar correlações básicas para tornar a EDA interessante
    # Se tem histórico familiar, aumenta chance de obesidade (simulado via peso)
    df.loc[df['family_history_with_overweight'] == 'yes', 'Weight'] += np.random.uniform(10, 30, size=len(df[df['family_history_with_overweight'] == 'yes']))
    
    # Definir NObeyesdad baseado no peso para simular a realidade do dataset
    bins = [0, 50, 70, 85, 100, 120, 140, 250]
    df['NObeyesdad'] = pd.cut(df['Weight'], bins=bins, labels=obesity_levels)
    
    return df

if __name__ == "__main__":
    df = generate_obesity_data()
    df.to_csv('/home/ubuntu/obesity-prediction/data/obesity.csv', index=False)
    print("Dataset gerado com sucesso em data/obesity.csv")
