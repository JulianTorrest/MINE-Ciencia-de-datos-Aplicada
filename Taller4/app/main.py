

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn import tree

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay

data_pre = pd.read_json('https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/data'
                        '/DataSet_Prediccion.json')
data_train_1 = pd.read_json(
    'https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/data/DataSet_Entrenamiento_v1.json')

st.title('Taller 4, churn Rate')


def load_data():
    uploaded_file = st.file_uploader(label='upload dataset for training')
    # data_train = pd.read_json(
    #   'https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/data/DataSet_Entrenamiento_v2.json')
    if uploaded_file is not None:
        data_train = uploaded_file.getvalue()
        return data_train


def load_pred():
    uploaded_file = st.file_uploader(label='upload dataset for training')
    # data_pred = pd.read_json('https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/data'
    # '/DataSet_Prediccion.json')
    if uploaded_file is not None:
        data_pred = uploaded_file.getvalue()
        return data_pred


def cleaning(dataset):
    # se limpia la columna TotalChares que tiene problemas con valores 0
    dataset.loc[dataset["tenure"] == 0, "TotalCharges"] = "0"
    dataset['TotalCharges'].astype(float)
    # se asocian por tipo las columnas
    objetivo = dataset["Churn"]
    numericas = dataset[["MonthlyCharges", "tenure", "SeniorCitizen", "TotalCharges"]]
    gender = dataset["gender"]
    categoricas_1 = dataset[['Partner', 'PhoneService', 'PaperlessBilling', 'Dependents']]
    categoricas_2 = dataset[[
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]]
    excluidas = dataset[["customerID", "MultipleLines"]]

    # se hace la transformación
    objetivo = objetivo.replace(['No', 'Yes'], [0, 1])
    gender = gender.replace(['Female', 'Male'], [0, 1])
    categoricas_1 = categoricas_1.replace(['No', 'Yes'], [0, 1])
    categoricas_2 = pd.get_dummies(categoricas_2)
    numericas = numericas.astype(float)

    # se construye todo el dataset limpio de nuevo
    clean_dataset = pd.DataFrame().join([objetivo, gender, categoricas_1, categoricas_2, numericas], how="outer")
    return clean_dataset


def cleaning_1(dataset):
    # se limpia la columna TotalChares que tiene problemas con valores 0
    dataset.loc[dataset["tenure"] == 0, "TotalCharges"] = "0"
    dataset['TotalCharges'].astype(float)
    # se asocian por tipo las columnas
    numericas = dataset[["MonthlyCharges", "tenure", "SeniorCitizen", "TotalCharges"]]
    gender = dataset["gender"]
    categoricas_1 = dataset[['Partner', 'PhoneService', 'PaperlessBilling', 'Dependents']]
    categoricas_2 = dataset[[
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]]
    excluidas = dataset[["customerID", "MultipleLines"]]

    # se hace la transformación
    gender = gender.replace(['Female', 'Male'], [0, 1])
    categoricas_1 = categoricas_1.replace(['No', 'Yes'], [0, 1])
    categoricas_2 = pd.get_dummies(categoricas_2)
    numericas = numericas.astype(float)

    # se construye todo el dataset limpio de nuevo
    clean_dataset = pd.DataFrame().join([gender, categoricas_1, categoricas_2, numericas], how="outer")
    return clean_dataset


def get_final_pred_mv0(dataset, model):
    # se limpia para que pueda ser ingerido por el modelo
    clean_df3 = cleaning_1(dataset)

    # se hace la predicción con el primer modelo
    df_predicted = pd.DataFrame(model.predict(clean_df3)).replace([0, 1], ['No', 'Yes'])
    df_precited_proba = pd.DataFrame(model.predict_proba(clean_df3)[:, 1])
    df_predicted["proba"] = df_precited_proba
    df_predicted.columns = ['Churn', 'Proba']

    return df_predicted


def reentrenamiento(df1, df2):
    clean_df1 = cleaning(df1)
    clean_df2 = cleaning(df2)
    rtrain_df = pd.concat([clean_df1, clean_df2], axis=0)

    # dividir el X y Y
    X = rtrain_df.drop("Churn", axis=1)
    Y = rtrain_df["Churn"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=33)

    scaler = StandardScaler()

    # se hace el pipeline con la regresión logística
    logistic = LogisticRegression(max_iter=1000, tol=0.1, class_weight='balanced', multi_class='multinomial',
                                  random_state=33)
    pipe = Pipeline(steps=[("scaler", scaler), ("polynomial", PolynomialFeatures()), ("logistic", logistic)])
    param_grid = {
        "polynomial__degree": [1, 2],
        "polynomial__interaction_only": [True, False],
        "polynomial__include_bias": [True, False],
        "logistic__penalty": ['l2', 'elasticnet'],
        "logistic__solver": ['liblinear', 'saga'],
    }

    # se busca el mejor modelo y regresa el score
    logistic_rtmodel = GridSearchCV(pipe, param_grid, n_jobs=2, scoring='roc_auc', cv=5).fit(X, Y)
    # score_logistic = roc_auc_score(Y_test, logistic_model.predict_proba(X_test)[:, 1])

    # v_score_logistic = cross_val_score(logistic_model, X_train, Y_train, cv=5, scoring='roc_auc').mean()

    return logistic_rtmodel


if st.checkbox('check for use first model'):
    # load model
    url = 'https://raw.githubusercontent.com/JulianTorrest/MINE-Ciencia-de-datos-Aplicada/main/Taller4/model/my_model.pkl'
    response = requests.get(url)
    open("my_model.pkl", "wb").write(response.content)
    best_model = joblib.load("my_model.pkl")
    uploaded_file = st.file_uploader(label='upload dataset for prediction')
    final_d = ''
    if uploaded_file is not None:
        data_predi = pd.read_json(uploaded_file.getvalue())
        prediction = get_final_pred_mv0(data_predi, best_model)
        print("final prediction", prediction)
        final_d = st.dataframe(prediction)
        st.write(f"Your churn results: {final_d}")

if st.button('Make Prediction with new model'):
    data_2_train = load_data()
    # data_2_train = cleaning(data_2_train)
    model_2 = reentrenamiento(data_2_train, data_train_1)
    prediction2 = get_final_pred_mv0(data_pre, model_2)
    # print("final prediction", np.squeeze(prediction2, -1))
    final_d2 = pd.DataFrame.to_string(prediction2)
    st.write(f"Your churns: {final_d2}")
