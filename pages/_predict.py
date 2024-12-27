import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Model yükleme fonksiyonu
def load_model():
    try:
        model_path = os.path.join('models', 'model.pkl')
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sayfanın başında model yükle
model = load_model()

def predict_page():
    st.title("Churn Prediction")
    
    try:
        # Form içine alalım
        with st.form("prediction_form"):
            # Input alanları
            col1, col2 = st.columns(2)
            
            with col1:
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
                balance = st.number_input("Balance", min_value=0.0, value=50000.0)
                num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
                
            with col2:
                has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
                active_member = st.selectbox("Is Active Member", ["Yes", "No"])
                estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
                geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
                gender = st.selectbox("Gender", ["Female", "Male"])
            
            # Tahmin butonu
            submit_button = st.form_submit_button("Predict")
            
            if submit_button:
                # Input verilerini DataFrame'e dönüştür
                input_data = {
                    'CreditScore': credit_score,
                    'Age': age,
                    'Tenure': tenure,
                    'Balance': balance,
                    'NumOfProducts': num_products,
                    'HasCrCard': 1 if has_card == "Yes" else 0,
                    'IsActiveMember': 1 if active_member == "Yes" else 0,
                    'EstimatedSalary': estimated_salary,
                    'Geography': geography,
                    'Gender': gender
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Tahmin işlemi
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                
                # Sonuçları göster
                st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
                st.info(f"Probability of Churn: {probability[1]:.2%}")
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug info:", e)

if __name__ == "__main__":
    predict_page()
