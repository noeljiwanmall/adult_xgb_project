import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Income Prediction", layout="centered")
st.title("🏡 Predict Income Bracket (>$50K)")

st.markdown("Fill in the details below and click **Predict** to see if the person earns over $50K.")

# Define input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])
    fnlwgt = st.number_input("Fnlwgt", min_value=10000, max_value=1000000, value=200000)
    education = st.selectbox("Education", [
        "Bachelors", "HS-grad", "Some-college", "Assoc-acdm", "Assoc-voc",
        "Masters", "Doctorate", "Prof-school", "Preschool", "12th"
    ])
    education_num = st.number_input("Education Num", min_value=1, max_value=16, value=13)
    marital_status = st.selectbox("Marital Status", [
        "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent"
    ])
    occupation = st.selectbox("Occupation", [
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Machine-op-inspct",
        "Sales", "Tech-support", "Craft-repair", "Transport-moving", "Other-service"
    ])
    relationship = st.selectbox("Relationship", [
        "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
    ])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada"])

    submit = st.form_submit_button("Predict")

# Handle prediction request
if submit:
    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }])

    payload = {
        "dataframe_split": {
            "columns": input_data.columns.tolist(),
            "data": input_data.values.tolist()
        }
    }

    try:
        is_docker = os.path.exists("/.dockerenv")

        default_url = "http://mlflow_server:5002/invocations" if is_docker else "http://localhost:5002/invocations"
        

        response = requests.post(
            # url="http://127.0.0.1:5002/invocations",

            url = os.getenv("MODEL_SERVER_URL", default_url),
            # url = "http://localhost:5002/invocations",
            # url = os.getenv("MODEL_SERVER_URL", "http://mlflow_server:5002/invocations"),
            headers={"Content-Type": "application/json"},
            json=payload
        )
        if response.status_code == 200:
            result = response.json()
            # prediction = result["predictions"][0]

            if "predictions" in result and isinstance(result["predictions"], list):
                prediction = result["predictions"][0]
            else:
                st.error("⚠️ Invalid response format from model server.")


            if prediction == 1:
                st.success("✅ Prediction: Income > $50K")
            else:
                st.info("🔍 Prediction: Income ≤ $50K")
        else:
            st.error(f"❌ Server error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Request failed: {e}")
