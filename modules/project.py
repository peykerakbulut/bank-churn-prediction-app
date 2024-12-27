import streamlit as st

def home_page():
    # Başlık
    st.title('Bank Customer Churn Prediction APP')

    # İçerik
    st.markdown("- Customer loyalty is of critical importance for banks and financial institutions. Customer churn not only results in revenue loss but also incurs higher costs for acquiring new customers. Therefore, predicting the likelihood of customers leaving the bank and developing preventive strategies is an essential step to enhance the competitive edge of banks.")
    st.markdown("- In this project, we aim to predict customer churn rates in banks using machine learning techniques. By analyzing past customer behaviors and their usage levels of the bank's services, we strive to identify which customers are more likely to leave the bank.")
    st.image("images/data_processing.png", width=500)

# Fonksiyonu direkt çağırmayalım, home.py'dan çağıracağız
