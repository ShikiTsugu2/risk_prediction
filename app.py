import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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

# Chargement des données pour analyse visuelle
data = pd.read_csv('credit_risk_dataset.csv', sep=';')

# Affichage de la distribution des variables
st.subheader("Répartition des ages des clients")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data['person_age'], kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Affichage de la heatmap de corrélation
st.subheader("Corrélation entre les Variables Numériques")
numeric_data = data.select_dtypes(include=['float64', 'int64'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Bouton de prédiction
# Le modèle prédit le risque de crédit du client soit loan_status = 1 (risqué) ou loan_status = 0 (non risqué)
def predict_risk():
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = "Client Risqué" if prediction[0] == 1 else "Client Non Risqué"
    return result

if st.button("Prédire"):
    result = predict_risk()
    st.write("### Résultat :", result)