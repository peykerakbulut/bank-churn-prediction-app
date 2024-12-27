import streamlit as st
import pandas as pd
import joblib
import os

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
            # Input alanları...
            
            # Tahmin butonu
            submit_button = st.form_submit_button("Predict")
            
            if submit_button:
                # Tahmin işlemi
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                
                # Sonuçları göster
                st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
                st.info(f"Probability: {probability[1]:.2%}")
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug info:", e)

if __name__ == "__main__":
    predict_page()
