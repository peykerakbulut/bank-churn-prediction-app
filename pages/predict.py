import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import shap
from datetime import date, datetime

st.header("🏦 Customer Churn Prediction")
st.markdown("**Choose the customer features to predict churn probability!**")

# Müşteri özelliklerini kullanıcıdan alma
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_credit_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])
card_type = st.selectbox("Card Type", ['SILVER', 'GOLD', 'PLATINUM', 'DIAMOND'])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict Churn Probability"):
    # Veriyi hazırla
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_credit_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'Card Type': card_type,
        'EstimatedSalary': estimated_salary
    }
    
    input_df = pd.DataFrame([data])
    
    # Kategorik değişkenleri dönüştür
    input_df['Geography'] = input_df['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
    input_df['Gender'] = input_df['Gender'].map({'Female': 0, 'Male': 1})
    input_df['Card Type'] = input_df['Card Type'].map({'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3})
    
    # Model ve gerekli dosyaları yükle
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Tahmin yap
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Sonuçları göster
    st.header("Prediction Results")
    
    # Churn olasılığını görselleştir
    churn_probability = prediction_proba[0][1] * 100
    
    # Gösterge oluştur
    fig = plt.figure(figsize=(10, 3))
    plt.barh(['Churn Probability'], [churn_probability], color='#2E86C1')
    plt.xlim(0, 100)
    plt.title('Churn Probability')
    
    for i, v in enumerate([churn_probability]):
        plt.text(v + 1, i, f'{v:.1f}%', va='center')
    
    st.pyplot(fig)
    
    # Tahmin sonucunu göster
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Churn!")
    else:
        st.success("✅ Low Risk of Churn")
    
    # Müşteri özelliklerini tablo olarak göster
    st.subheader("Customer Features")
    result_df = pd.DataFrame({
        'Feature': data.keys(),
        'Value': data.values()
    })
    st.table(result_df)
    
    # SHAP değerlerini hesapla ve göster
    try:
        with open('explainer.pkl', 'rb') as file:
            explainer = pickle.load(file)
            
        shap_values = explainer(input_df)
        
        st.subheader("Feature Impact Analysis")
        fig_shap = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        st.pyplot(fig_shap)
    except Exception as e:
        st.warning("SHAP analysis is not available at the moment.")

# Batch prediction bölümü
st.header("📥 Batch Prediction")
uploaded_file = st.file_uploader("**Upload CSV file for batch prediction**", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    
    # Kategorik değişkenleri dönüştür
    category_mappings = {
        'Geography': {'France': 0, 'Germany': 1, 'Spain': 2},
        'Gender': {'Female': 0, 'Male': 1},
        'Card Type': {'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3}
    }
    
    for col, mapping in category_mappings.items():
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].map(mapping)
    
    # Tahminleri yap
    batch_pred = model.predict(batch_df)
    batch_pred_proba = model.predict_proba(batch_df)
    
    # Sonuçları DataFrame'e ekle
    results_df = batch_df.copy()
    results_df['Churn_Prediction'] = batch_pred
    results_df['Churn_Probability'] = batch_pred_proba[:, 1]
    
    st.subheader("Batch Prediction Results")
    st.dataframe(results_df)
    
    # Download butonu
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )