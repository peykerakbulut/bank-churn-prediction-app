import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Başlık
st.title("About Dataset")

# HTML dosyasını oku
with open('customer_churn_report.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# HTML'i doğrudan components.html ile göster
components.html(html_content, height=800, scrolling=True)

