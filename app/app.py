"""
Aplicação de Apoio ao Diagnóstico de Obesidade
----------------------------------------------
Esta aplicação utiliza um modelo de Machine Learning para prever o nível de
obesidade com base em dados biométricos e hábitos de vida.

Interface traduzida para Português (Brasil).
"""

import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Adicionar o diretório src ao path para permitir importações se necessário
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def load_artifacts():
    """Carrega o modelo e o label encoder salvos."""
    model_path = os.path.join('models', 'best_model_diagnostic.pkl')
    le_path = os.path.join('models', 'label_encoder.pkl')
    
    if not os.path.exists(model_path):
        model_path = os.path.join('..', 'models', 'best_model_diagnostic.pkl')
        le_path = os.path.join('..', 'models', 'label_encoder.pkl')
        
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le

# --- Dicionários de Mapeamento para Internacionalização ---

# Mapeamento de Saída (Classes de Obesidade)
output_labels = {
    "Insufficient_Weight": "Abaixo do peso",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso (Grau I)",
    "Overweight_Level_II": "Sobrepeso (Grau II)",
    "Obesity_Type_I": "Obesidade Grau I",
    "Obesity_Type_II": "Obesidade Grau II",
    "Obesity_Type_III": "Obesidade Grau III"
}

# Mapeamento de Opções de Entrada
binary_labels = {"yes": "Sim", "no": "Não"}
gender_labels = {"Male": "Masculino", "Female": "Feminino"}
frequency_labels = {
    "no": "Nunca",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre"
}
mtrans_labels = {
    "Public_Transportation": "Transporte Público",
    "Walking": "Caminhada",
    "Automobile": "Automóvel",
    "Motorbike": "Motocicleta",
    "Bike": "Bicicleta"
}

def main():
    st.set_page_config(page_title="Diagnóstico de Obesidade", layout="wide")
    
    st.title("🩺 Sistema de Apoio ao Diagnóstico de Obesidade")
    st.markdown("""
    Esta ferramenta auxilia profissionais de saúde na classificação do nível de obesidade 
    com base em parâmetros biométricos e comportamentais.
    """)
    
    try:
        model, le = load_artifacts()
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}. Certifique-se de que o treinamento foi realizado.")
        return

    st.sidebar.header("📋 Dados do Paciente")
    
    # --- Inputs Biométricos ---
    st.sidebar.subheader("Biometria")
    gender = st.sidebar.selectbox("Gênero", options=list(gender_labels.keys()), format_func=lambda x: gender_labels[x])
    age = st.sidebar.number_input("Idade", min_value=14, max_value=100, value=25)
    height = st.sidebar.number_input("Altura (m)", min_value=1.40, max_value=2.50, value=1.70, step=0.01)
    weight = st.sidebar.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
    
    # --- Histórico e Hábitos Alimentares ---
    st.sidebar.subheader("Hábitos Alimentares")
    family_history = st.sidebar.selectbox("Histórico Familiar de Sobrepeso?", options=list(binary_labels.keys()), format_func=lambda x: binary_labels[x])
    favc = st.sidebar.selectbox("Consumo frequente de alimentos calóricos?", options=list(binary_labels.keys()), format_func=lambda x: binary_labels[x])
    caec = st.sidebar.selectbox("Consumo de alimentos entre refeições", options=list(frequency_labels.keys()), format_func=lambda x: frequency_labels[x], index=1)
    fcvc = st.sidebar.slider("Frequência de consumo de vegetais (1-3)", 1.0, 3.0, 2.0, step=0.1)
    ncp = st.sidebar.slider("Número de refeições principais (1-4)", 1.0, 4.0, 3.0, step=0.1)
    calc = st.sidebar.selectbox("Consumo de álcool", options=list(frequency_labels.keys()), format_func=lambda x: frequency_labels[x], index=0)
    
    # --- Estilo de Vida ---
    st.sidebar.subheader("Estilo de Vida")
    smoke = st.sidebar.selectbox("Fumante?", options=list(binary_labels.keys()), format_func=lambda x: binary_labels[x], index=1)
    ch2o = st.sidebar.slider("Consumo diário de água (1-3)", 1.0, 3.0, 2.0, step=0.1)
    scc = st.sidebar.selectbox("Monitora ingestão calórica?", options=list(binary_labels.keys()), format_func=lambda x: binary_labels[x], index=1)
    faf = st.sidebar.slider("Frequência de atividade física (0-3)", 0.0, 3.0, 1.0, step=0.1)
    tue = st.sidebar.slider("Tempo de uso de eletrônicos (0-2)", 0.0, 2.0, 1.0, step=0.1)
    mtrans = st.sidebar.selectbox("Meio de transporte habitual", options=list(mtrans_labels.keys()), format_func=lambda x: mtrans_labels[x])

    # --- Organização dos Dados para Inferência ---
    # Os valores enviados ao DataFrame são as chaves originais (ex: "yes", "Male")
    input_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }
    
    df_input = pd.DataFrame([input_data])
    
    # --- Exibição e Predição ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Resumo dos Dados")
        # Para exibição, traduzimos os valores do DataFrame
        df_display = df_input.copy().T.rename(columns={0: "Valor"})
        st.dataframe(df_display)
        
    with col2:
        st.subheader("Resultado da Análise")
        if st.button("Realizar Predição"):
            with st.spinner("Processando..."):
                # A pipeline recebe o DataFrame com valores originais
                prediction_encoded = model.predict(df_input)
                prediction_raw = le.inverse_transform(prediction_encoded)[0]
                
                # Tradução da saída para exibição
                prediction_translated = output_labels.get(prediction_raw, prediction_raw)
                
                st.success(f"Nível de Obesidade Previsto: **{prediction_translated}**")
                
                st.info("""
                **Nota Importante:** Este resultado é gerado por um modelo estatístico e deve ser 
                interpretado por um profissional de saúde qualificado como parte de uma avaliação clínica completa.
                """)
                
                # Dica visual baseada no resultado original
                if "Obesity" in prediction_raw:
                    st.warning("Atenção: O perfil indica necessidade de intervenção clínica e nutricional.")
                elif "Overweight" in prediction_raw:
                    st.info("O perfil indica tendência ao sobrepeso. Recomenda-se monitoramento de hábitos.")
                else:
                    st.balloons()
                    st.write("O perfil está dentro dos parâmetros de normalidade ou abaixo do peso.")

if __name__ == "__main__":
    main()
