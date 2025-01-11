import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def evaluation_page():
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
    
    st.title("Model Evaluation")
    
    try:
        # Model ve veriyi yükle
        with open('models/model.pkl', 'rb') as file:
            model = pickle.load(file)
            
        # Feature mapping'i yükle
        with open('models/feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
            
        # Test verilerini yükle
        df = pd.read_csv('data/customer_churn_records.csv')
        X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Complain'], axis=1)
        y = df['Exited']
        
        # Feature isimlerini düzenle
        X = X.rename(columns=feature_names)
        
        # Kategorik değişkenleri dönüştür
        categorical_columns = ['Country', 'Gender', 'Card Type']
        for column in categorical_columns:
            if column == 'Country':
                X[column] = X[column].map({'France': 0, 'Germany': 1, 'Spain': 2})
            elif column == 'Gender':
                X[column] = X[column].map({'Female': 0, 'Male': 1})
            elif column == 'Card Type':
                X[column] = X[column].map({'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3})
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        conf_matrix_df = pd.DataFrame(conf_matrix,
                                    columns=["Not Churn", "Churn"],
                                    index=["Not Churn", "Churn"])

        fig_cm = px.imshow(
            conf_matrix_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#2E86C1", "#ffbd59"],
            labels={"color": "Count"}
        )
        
        fig_cm.update_layout(title="Confusion Matrix")
        fig_cm.update_xaxes(title_text="Prediction")
        fig_cm.update_yaxes(title_text="Actual")

        st.plotly_chart(fig_cm, use_container_width=True)

        # Evaluation Metrics
        st.subheader("Evaluation Metrics")
        metrics = {
            "Not Churn": {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, pos_label=0),
                "Recall": recall_score(y_test, y_pred, pos_label=0),
                "F1 Score": f1_score(y_test, y_pred, pos_label=0),
                "AUC Score": roc_auc_score(y_test, y_pred_proba)
            },
            "Churn": {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, pos_label=1),
                "Recall": recall_score(y_test, y_pred, pos_label=1),
                "F1 Score": f1_score(y_test, y_pred, pos_label=1),
                "AUC Score": roc_auc_score(y_test, y_pred_proba)
            }
        }
        
        for group in ["Not Churn", "Churn"]:
            df_metrics = pd.DataFrame({
                "Customer Group": [group],
                "Accuracy": [f"{metrics[group]['Accuracy']:.3f}"],
                "Precision": [f"{metrics[group]['Precision']:.3f}"],
                "Recall": [f"{metrics[group]['Recall']:.3f}"],
                "F1 Score": [f"{metrics[group]['F1 Score']:.3f}"],
                "AUC Score": [f"{metrics[group]['AUC Score']:.3f}"]
            })
            st.table(df_metrics)

        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        fig_roc = plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='#2E86C1', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_roc)

        # Interactive Threshold Selection
        st.subheader("Interactive Threshold Selection")
        
        # Session state kullan
        if 'threshold' not in st.session_state:
            st.session_state.threshold = 0.5
            
        threshold = st.slider("Select Threshold", 0.0, 1.0, st.session_state.threshold, 0.01, 
                            key='threshold_slider')
        st.session_state.threshold = threshold
        
        # Seçilen threshold'a göre tahminleri güncelle
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        # Yeni confusion matrix ve metrikleri hesapla
        conf_matrix = confusion_matrix(y_test, y_pred_threshold)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)
        
        # Sonuçları göster
        st.write("\nResults with selected threshold:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Precision", f"{report['1']['precision']:.3f}")
            st.metric("Recall", f"{report['1']['recall']:.3f}")
        
        with col2:
            st.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
            st.metric("Accuracy", f"{report['accuracy']:.3f}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Debug info:", e)
        st.write("Please make sure all required files exist in the correct locations:")
        st.write("- models/model.pkl")
        st.write("- models/feature_names.pkl")
        st.write("- data/customer_churn_records.csv")
