import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

data_pre = pd.read_json('https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/data'
                        '/DataSet_Prediccion.json')

st.title('Taller 4, Tasa Churn')


def load_data():
    uploaded_file = st.file_uploader(label='Cargar datos')
    data_train = pd.read_json(
        'https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/data/DataSet_Entrenamiento_v2.json')
    if uploaded_file is not None:
        data_train = uploaded_file.getvalue()
    return data_train


if st.checkbox('Verificar modelo'):
    # load model
    url = 'ttps://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/model/my_model.pkl'
    response = requests.get(url)
    open("my_model.pkl", "wb").write(response.content)
    best_model = joblib.load("my_model.pkl")
if st.button('Make Prediction'):
    inputs = data_pre
    prediction = best_model.predict(inputs)
    print("final prediction", np.squeeze(prediction, -1))
    final_d = np.array2string(prediction)
    st.write(f"Your churn: {final_d}g")
load_data()

if st.button('Realizar predicción'):
    inputs = data_pre
    prediction = best_model.predict(inputs)
    print("final prediction", np.squeeze(prediction, -1))
    final_d = np.array2string(prediction)
    st.write(f"Your fares: {final_d}g")