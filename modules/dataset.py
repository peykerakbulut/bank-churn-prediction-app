import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

def dataset_page():
    # Başlık
    st.title("About Dataset")
    
    # Container genişliğini artır
    st.markdown("""
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # HTML dosyasını oku
    with open('static/customer_churn_report.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # HTML'i doğrudan components.html ile göster
    components.html(html_content, height=800, scrolling=True)

