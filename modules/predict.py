import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

def load_model():
    try:
        model_path = os.path.join('models', 'model.pkl')
        label_encoders_path = os.path.join('models', 'label_encoders.pkl')
        
        # Model dosyasının varlığını kontrol et
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None
            
        # Dosyaları yükle
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(label_encoders_path, 'rb') as file:
            label_encoders = pickle.load(file)
            
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_page():
    st.title("Churn Prediction")
    
    try:
        # Model ve encoder'ları yükle
        model, label_encoders = load_model()
        
        if model is not None and label_encoders is not None:
            # Debug için model özelliklerini yazdır
            st.write("Model features:", model.feature_names_in_)
            
            # Input alanları
            col1, col2 = st.columns(2)
            
            # Form dışında input alanlarını oluştur
            with col1:
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
                balance = st.number_input("Balance", min_value=0.0, value=50000.0)
                num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
                satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
                points_earned = st.number_input("Points Earned", min_value=0, value=100)
                
            with col2:
                has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
                active_member = st.selectbox("Is Active Member", ["Yes", "No"])
                estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
                geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
                gender = st.selectbox("Gender", ["Female", "Male"])
                card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
            
            # Predict butonu
            if st.button("Predict", type="primary"):
                # Input verilerini DataFrame'e dönüştür
                input_data = pd.DataFrame([{
                    'Credit Score': credit_score,
                    'Country': geography,
                    'Gender': gender,
                    'Age': age,
                    'Tenure': tenure,
                    'Balance': balance,
                    'Number of Products': num_products,
                    'Has Credit Card': 1 if has_card == "Yes" else 0,
                    'Is Active Member': 1 if active_member == "Yes" else 0,
                    'Estimated Salary': estimated_salary,
                    'Satisfaction Score': satisfaction_score,
                    'Card Type': card_type,
                    'Points Earned': points_earned
                }])
                
                input_df = input_data.copy()  # input_data'yı input_df'e kopyala
                
                # Kategorik değişkenleri dönüştür
                categorical_columns = ['Country', 'Gender', 'Card Type']
                for column in categorical_columns:
                    input_df[column] = label_encoders[column].transform(input_df[column])
                
                # Debug için input özelliklerini yazdır
                st.write("Input features:", input_df.columns.tolist())
                
                # Tahmin işlemi
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                
                # Sonuç container'ı
                result_container = st.container()
                
                # Sonuçları göster
                with result_container:
                    if prediction == 1:
                        st.error("⚠️ High Risk of Churn!")
                    else:
                        st.success("✅ Low Risk of Churn")
                    
                    st.info(f"Probability of Churn: {probability[1]:.2%}")
                    
                    # Detaylı olasılıklar
                    st.write("\nDetailed Probabilities:")
                    prob_df = pd.DataFrame({
                        'Outcome': ['Not Churn', 'Churn'],
                        'Probability': [f"{probability[0]:.2%}", f"{probability[1]:.2%}"]
                    })
                    st.table(prob_df)
                    
        else:
            st.warning("Please make sure all required files exist in the correct locations:")
            st.write("- models/model.pkl")
            st.write("- models/label_encoders.pkl")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug info:", e)
