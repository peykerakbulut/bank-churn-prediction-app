import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Renk teması
color_theme = ['#2E86C1', '#ffbd59']  # Mavi ve Sarı

def charts_page():
    st.title("Customer Churn Analysis")
    
    # Veriyi oku
    df = pd.read_csv("data/customer_churn_records.csv")
    
    # Filters bölümünü sayfanın üst kısmına ekle
    with st.expander("📊 Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter seçenekleri
            age_range = st.slider("Age Range", 
                                int(df['Age'].min()), 
                                int(df['Age'].max()), 
                                (20, 80))
            balance_range = st.slider("Balance Range", 
                                    float(df['Balance'].min()), 
                                    float(df['Balance'].max()), 
                                    (0.0, 250000.0))
            
        with col2:
            # Diğer filtreler
            geography = st.multiselect("Geography",
                                     df['Geography'].unique(),
                                     default=df['Geography'].unique())
            card_types = st.multiselect("Card Type",
                                      df['Card Type'].unique(),
                                      default=df['Card Type'].unique())
    
    # Filtreleme işlemi
    filtered_df = df[
        (df['Age'].between(age_range[0], age_range[1])) &
        (df['Balance'].between(balance_range[0], balance_range[1])) &
        (df['Geography'].isin(geography)) &
        (df['Card Type'].isin(card_types))
    ]
    
    # Filtrelenmiş veri seti hakkında bilgi
    st.info(f"Showing data for {len(filtered_df)} customers (out of {len(df)} total)")
    
    # 1. Churn Dağılımı (Pie Chart)
    st.subheader("Customer Churn Distribution")
    fig_churn = px.pie(filtered_df, names='Exited', 
                       title='Customer Churn Rate',
                       color_discrete_sequence=color_theme)
    st.plotly_chart(fig_churn)

    # 2. Ülkelere Göre Churn (Bar Chart)
    st.subheader("Churn by Geography")
    churn_by_country = filtered_df.groupby(['Geography', 'Exited']).size().unstack()
    fig_geo = px.bar(churn_by_country, 
                     title='Churn Distribution by Country',
                     barmode='group',
                     color_discrete_sequence=color_theme)
    st.plotly_chart(fig_geo)

    # 3. Yaş ve Kredi Skoru İlişkisi (Scatter Plot)
    st.subheader("Age vs Credit Score by Churn Status")
    fig_scatter = px.scatter(filtered_df, 
                            x='Age', 
                            y='CreditScore',
                            color='Exited',
                            title='Age vs Credit Score',
                            color_discrete_sequence=color_theme)
    st.plotly_chart(fig_scatter)

    # 4. Kart Tipine Göre Churn (Grouped Bar Chart)
    st.subheader("Churn by Card Type")
    card_churn = pd.crosstab(filtered_df['Card Type'], filtered_df['Exited'])
    fig_card = px.bar(card_churn, 
                      title='Churn Distribution by Card Type',
                      barmode='group',
                      color_discrete_sequence=color_theme)
    st.plotly_chart(fig_card)

    # 5. Müşteri Memnuniyeti ve Churn İlişkisi (Box Plot)
    st.subheader("Satisfaction Score Distribution by Churn Status")
    fig_box = px.box(filtered_df, 
                     x='Exited', 
                     y='Satisfaction Score',
                     title='Satisfaction Score vs Churn',
                     color='Exited',
                     color_discrete_sequence=color_theme)
    st.plotly_chart(fig_box)

    # 6. Balance Distribution (Histogram)
    st.subheader("Balance Distribution by Churn Status")
    fig_hist = px.histogram(filtered_df, 
                           x='Balance',
                           color='Exited',
                           title='Balance Distribution',
                           marginal='box',
                           color_discrete_sequence=color_theme)
    st.plotly_chart(fig_hist)

    # 7. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    correlation = filtered_df[numeric_cols].corr()
    fig_corr = px.imshow(correlation,
                         title='Correlation Matrix',
                         color_continuous_scale='Blues')  # Mavi tonlarında heatmap
    st.plotly_chart(fig_corr)

    # 8. Point Earned Distribution
    st.subheader("Point Earned Distribution by Card Type")
    fig_points = px.box(filtered_df, 
                       x='Card Type', 
                       y='Point Earned',
                       color='Exited',
                       title='Point Distribution by Card Type',
                       color_discrete_sequence=color_theme)
    st.plotly_chart(fig_points)

    # Custom CSS to style the charts
    st.markdown("""
        <style>
        .plot-container {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
