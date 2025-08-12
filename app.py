import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

MODEL_PATH = os.path.join("model", "best_model.pkl")
SKIN_MODEL_PATH = os.path.join("model", "skin_thickness_estimator.pkl")
DATA_PATH = os.path.join("data", "diabetes.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

df = load_data()

model = None
skin_model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.warning("Trained diabetes prediction model not found. Please add model/best_model.pkl")

if os.path.exists(SKIN_MODEL_PATH):
    skin_model = load_model(SKIN_MODEL_PATH)
else:
    st.warning("Skin thickness estimation model not found. Please add model/skin_thickness_estimator.pkl")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance", "About"])

if page == "Home":
    st.title("Diabetes Risk Predictor")
    st.write("Interactive ML app to predict diabetes risk using the Pima Indians Diabetes dataset.")
    st.markdown("""
    **Features**
    - Data exploration and interactive filters
    - Visualizations (histograms, boxplots, correlation heatmap)
    - Predict whether a person is likely diabetic with probability
    - Model performance metrics and confusion matrix
    """)
    st.write("Dataset shape:", df.shape)
    st.write("Class distribution:")
    st.bar_chart(df['Outcome'].value_counts())

    st.header("Simplified Diabetes Risk Prediction")
    st.write("Enter accessible health indicators below to get a prediction.")

    with st.form("simple_pred_form"):
        p1, p2, p3, p4 = st.columns(4)
        pregnancies = p1.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        age = p2.number_input("Age", min_value=1, max_value=120, value=int(df['Age'].median()))
        weight = p3.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
        height_cm = p4.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)

        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        st.write(f"Calculated BMI: **{bmi:.2f}**")

        p5 = st.number_input("Blood Pressure", min_value=0, max_value=200, value=int(df['BloodPressure'].median()))

        st.markdown("### Family History of Diabetes")
        parents_diabetes = st.radio("Do any of your parents have diabetes?", ("No", "Yes"))
        siblings_diabetes = st.radio("Do any of your siblings have diabetes?", ("No", "Yes"))
        grandparents_diabetes = st.radio("Do any grandparents or close relatives have diabetes?", ("No", "Yes"))

        dpf = 0.0
        if parents_diabetes == "Yes":
            dpf += 6
        if siblings_diabetes == "Yes":
            dpf += 5
        if grandparents_diabetes == "Yes":
            dpf += 4
        dpf = min(dpf, 1.0)

        st.write(f"Estimated Diabetes Pedigree Function (DPF): **{dpf:.2f}**")

        if skin_model is not None:
            skin_thickness = skin_model.predict([[bmi]])[0]
            skin_thickness = max(skin_thickness, 0)
            st.write(f"Estimated Skin Thickness: **{skin_thickness:.2f}**")
        else:
            skin_thickness = int(df['SkinThickness'].median())

        submitted = st.form_submit_button("Predict")

    if submitted:
        if model is None:
            st.error("Diabetes prediction model not loaded. Please add model/best_model.pkl")
        else:
            glucose = int(df['Glucose'].median())
            insulin = float(df['Insulin'].median())
            input_df = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": p5,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }])

            try:
                prob = model.predict_proba(input_df)[0][1]
                pred = model.predict(input_df)[0]
                st.metric(label="Diabetes Probability", value=f"{prob*100:.1f}%")
                if pred == 1:
                    st.error("Prediction: Likely diabetic (positive)")
                else:
                    st.success("Prediction: Not diabetic (negative)")
                st.info("Note: This prediction is based on a machine learning model and is not a medical diagnosis.")
            except Exception as e:
                st.exception(e)

if page == "Data Exploration":
    st.header("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.subheader("Sample rows")
    st.dataframe(df.sample(10, random_state=42))
    st.subheader("Interactive filtering")
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    glucose_range = st.slider("Filter by Glucose", int(df['Glucose'].min()), int(df['Glucose'].max()), (70, 140))
    filtered = df[(df['Age'].between(age_range[0], age_range[1])) & (df['Glucose'].between(glucose_range[0], glucose_range[1]))]
    st.write("Filtered shape:", filtered.shape)
    st.dataframe(filtered.head(50))

if page == "Visualizations":
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Glucose distribution")
        fig1 = px.histogram(df, x="Glucose", nbins=30, marginal="box", title="Glucose distribution")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("BMI vs Age")
        fig2 = px.scatter(df, x="Age", y="BMI", color="Outcome", title="BMI by Age (colored by Outcome)", trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df.corr()
    fig3, ax = plt.subplots(figsize=(8,6))
    ax = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(ax)
    st.pyplot(fig3)

    st.subheader("Outcome counts by Age group")
    df['age_group'] = pd.cut(df['Age'], bins=[20,30,40,50,60,100], labels=["21-30","31-40","41-50","51-60","60+"])
    counts = df.groupby(['age_group','Outcome']).size().reset_index(name='counts')
    fig4 = px.bar(counts, x='age_group', y='counts', color='Outcome', barmode='group', title="Outcome by age group")
    st.plotly_chart(fig4)

if page == "Prediction":
    st.header("Predict Diabetes Risk")
    st.write("Enter the health indicators below to get a prediction.")
    with st.form("pred_form"):
        p1, p2, p3, p4 = st.columns(4)
        pregnancies = p1.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = p2.number_input("Glucose", min_value=0, max_value=300, value=int(df['Glucose'].median()))
        blood_pressure = p3.number_input("BloodPressure", min_value=0, max_value=200, value=int(df['BloodPressure'].median()))
        # We remove SkinThickness input, will calculate using model
        p5, p6, p7, p8 = st.columns(4)
        insulin = p5.number_input("Insulin", min_value=0.0, max_value=1000.0, value=float(df['Insulin'].median()))
        bmi_weight = p6.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
        bmi_height_cm = p7.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
        dpf_parents = p8.radio("Parents have diabetes?", ("No", "Yes"))

        # Extra inputs for siblings and grandparents DPF on next line
        siblings_diabetes = st.radio("Siblings have diabetes?", ("No", "Yes"))
        grandparents_diabetes = st.radio("Grandparents/close relatives have diabetes?", ("No", "Yes"))

        submitted = st.form_submit_button("Predict")

    if submitted:
        if model is None or skin_model is None:
            st.error("Required models not loaded. Please add models in 'model' folder.")
        else:
            height_m = bmi_height_cm / 100
            bmi = bmi_weight / (height_m ** 2)
            st.write(f"Calculated BMI: **{bmi:.2f}**")

            # Calculate skin thickness using model
            skin_thickness = skin_model.predict([[bmi]])[0]
            skin_thickness = max(skin_thickness, 0)
            st.write(f"Estimated Skin Thickness: **{skin_thickness:.2f}**")

            # Calculate DPF from family history
            dpf = 0.0
            if dpf_parents == "Yes":
                dpf += 0.4
            if siblings_diabetes == "Yes":
                dpf += 0.4
            if grandparents_diabetes == "Yes":
                dpf += 0.4
            dpf = min(dpf, 1.0)
            st.write(f"Estimated Diabetes Pedigree Function (DPF): **{dpf:.2f}**")

            input_df = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": int(df['Age'].median())  # or ask user for Age input if preferred
            }])

            try:
                prob = model.predict_proba(input_df)[0][1]
                pred = model.predict(input_df)[0]
                st.metric(label="Diabetes Probability", value=f"{prob*100:.1f}%")
                if pred == 1:
                    st.error("Prediction: Likely diabetic (positive)")
                else:
                    st.success("Prediction: Not diabetic (negative)")
                st.info("Note: This prediction is based on a machine learning model and is not a medical diagnosis.")
            except Exception as e:
                st.exception(e)

if page == "Model Performance":
    st.header("Model Performance")
    metrics_path = os.path.join("model", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.write("Cross-validated metrics summary:")
        st.json(metrics)
    else:
        st.write("No metrics.json found. Showing quick evaluation on dataset (for demo only).")
        try:
            X = df.drop("Outcome", axis=1)
            y = df["Outcome"]
            ypred = model.predict(X)
            yprob = model.predict_proba(X)[:,1]
            st.write("Accuracy:", accuracy_score(y, ypred))
            st.write("Precision:", precision_score(y, ypred))
            st.write("Recall:", recall_score(y, ypred))
            st.write("F1:", f1_score(y, ypred))
            st.write("ROC AUC:", roc_auc_score(y, yprob))
            cm = confusion_matrix(y, ypred)
            st.write("Confusion Matrix:")
            st.write(cm)
        except Exception as e:
            st.error("Could not compute model performance: " + str(e))

if page == "About":
    st.header("About this project")
    st.markdown("""
    This project implements a Diabetes Risk Predictor using the Kaggle Pima Indians Diabetes dataset.
    The app shows EDA, interactive visualizations, allows user input for predictions, and displays model performance.
    - Model training script: `notebooks/model_training.py`
    - Model saved in: `model/best_model.pkl`
    """)
    st.write("Assignment rubric requirements checklist:")
    st.write("""
    - Title and description 
    - Sidebar navigation 
    - Data exploration section 
    - At least 3 charts 
    - Prediction UI + probability 
    - Model performance section 
    - Error handling for missing model/data 
    """)
