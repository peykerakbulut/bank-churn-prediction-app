import streamlit as st

def home_page():
    # Başlık
    st.title('Bank Customer Churn Prediction APP')

    # İki kolonlu düzen oluştur
    col1, col2 = st.columns([3, 2])  # 3:2 oranında kolonlar

    with col1:
        # Metin içeriği
        st.markdown("""
        Customer loyalty is of critical importance for banks and financial institutions. 
        Customer churn not only results in revenue loss but also incurs higher costs for acquiring new customers. 
        Therefore, predicting the likelihood of customers leaving the bank and developing preventive strategies 
        is an essential step to enhance the competitive edge of banks.
        
        In this project, we aim to predict customer churn rates in banks using machine learning techniques. 
        By analyzing past customer behaviors and their usage levels of the bank's services, 
        we strive to identify which customers are more likely to leave the bank.
        """)

    with col2:
        # Resim
        st.image("images/data-cleaning.png", use_container_width=True)

# Fonksiyonu direkt çağırmayalım, home.py'dan çağıracağız
