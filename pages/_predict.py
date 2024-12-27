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
