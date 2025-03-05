import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Charger le modèle et le scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Interface utilisateur Streamlit
st.title("Prédiction du Risque de Crédit")
st.write("Entrez les informations du client pour prédire le risque de crédit.")

# Charger les colonnes d'entrée
feature_names = ['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length']
input_data = {}

for col in feature_names:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Bouton de prédiction
# Le modèle prédit le risque de crédit du client soit loan_status = 1 (risqué) ou loan_status = 0 (non risqué)
def predict_risk():
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = "Client Risqué" if prediction[0] == 1 else "Client Non Risqué"
    return result

st.button("Prédire", on_click=predict_risk)

result = predict_risk()
st.write("### Résultat :", result)