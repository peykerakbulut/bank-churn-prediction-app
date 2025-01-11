import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

def load_model():
    try:
        model_path = os.path.join('models', 'model.pkl')
        label_encoders_path = os.path.join('models', 'label_encoders.pkl')
        feature_names_path = os.path.join('models', 'feature_names.pkl')
        
        # Dosyaları yükle
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(label_encoders_path, 'rb') as file:
            label_encoders = pickle.load(file)
        with open(feature_names_path, 'rb') as file:
            feature_names = pickle.load(file)
            
        return model, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def batch_page():
    st.title("Batch Customer Prediction")
    
    try:
        # Model ve gerekli dosyaları yükle
        model, label_encoders, feature_names = load_model()
        
        if model is not None and label_encoders is not None:
            # Dosya yükleme alanı
            st.write("### Upload Customer Data")
            st.write("Upload a CSV file containing customer information for batch prediction.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="The CSV file should contain customer information in the same format as the training data.")
            
            # Örnek CSV şablonu indirme
            sample_df = pd.DataFrame(columns=[
                'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Satisfaction Score', 'Card Type', 'Point Earned'
            ])
            
            st.download_button(
                label="📄 Download Template CSV",
                data=sample_df.to_csv(index=False),
                file_name="template.csv",
                mime="text/csv",
                help="Download a template CSV file with the required columns"
            )
            
            if uploaded_file is not None:
                try:
                    # CSV dosyasını oku
                    batch_df = pd.read_csv(uploaded_file)
                    
                    # Gereksiz sütunları kaldır
                    columns_to_drop = ['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Complain']
                    batch_df = batch_df.drop([col for col in columns_to_drop if col in batch_df.columns], axis=1)
                    
                    # Feature isimlerini düzenle
                    batch_df = batch_df.rename(columns=feature_names)
                    
                    # Veri önizleme
                    st.write("### Data Preview")
                    st.dataframe(batch_df.head())
                    
                    if st.button("Run Batch Analysis", type="primary"):
                        with st.spinner("Processing..."):
                            # Kategorik değişkenleri dönüştür
                            categorical_columns = ['Country', 'Gender', 'Card Type']
                            for column in categorical_columns:
                                batch_df[column] = label_encoders[column].transform(batch_df[column])
                            
                            # Tahminleri yap
                            batch_pred = model.predict(batch_df)
                            batch_pred_proba = model.predict_proba(batch_df)
                            
                            # Sonuçları DataFrame'e ekle
                            results_df = batch_df.copy()
                            results_df['Churn_Prediction'] = batch_pred
                            results_df['Churn_Probability'] = batch_pred_proba[:, 1]
                            
                            # Kategorik değişkenleri geri dönüştür
                            inverse_mappings = {
                                'Country': {0: 'France', 1: 'Germany', 2: 'Spain'},
                                'Gender': {0: 'Female', 1: 'Male'},
                                'Card Type': {0: 'SILVER', 1: 'GOLD', 2: 'PLATINUM', 3: 'DIAMOND'}
                            }
                            
                            for column, mapping in inverse_mappings.items():
                                results_df[column] = results_df[column].map(mapping)
                            
                            # Sonuçları göster
                            st.write("### Analysis Results")
                            st.dataframe(results_df)
                            
                            # İstatistikler
                            st.write("### Summary Statistics")
                            total_customers = len(results_df)
                            churn_count = results_df['Churn_Prediction'].sum()
                            churn_rate = churn_count / total_customers
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Customers", total_customers)
                            with col2:
                                st.metric("Predicted Churns", int(churn_count))
                            with col3:
                                st.metric("Churn Rate", f"{churn_rate:.1%}")
                            
                            # Download butonu
                            st.download_button(
                                label="📥 Download Analysis Results",
                                data=results_df.to_csv(index=False),
                                file_name="churn_analysis_results.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.write("Debug info:", e)
                    st.write("Please make sure your CSV file has the correct format and columns.")
        
        else:
            st.warning("Please make sure all required files exist in the correct locations.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Debug info:", e) 