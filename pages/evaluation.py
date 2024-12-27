import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# Model ve SHAP explainer'ı yükle
@st.cache_resource
def load_model_and_explainer():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('explainer.pkl', 'rb') as file:
            explainer = pickle.load(file)
        return model, explainer
    except Exception as e:
        st.error(f"Error loading model or explainer: {e}")
        return None, None

# Veriyi yükle ve hazırla
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("customer_churn_records.csv")
    
    # Feature names'i yükle
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
    
    X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Complain'], axis=1)
    # Feature isimlerini değiştir
    X = X.rename(columns=feature_names)
    y = df['Exited']
    
    # Label encoder'ları yükle
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    
    # Kategorik değişkenleri dönüştür
    categorical_columns = ['Country', 'Gender', 'Card Type']
    for column in categorical_columns:
        X[column] = label_encoders[column].transform(X[column])
    
    # Sayısal değerleri float64'e dönüştür
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_columns] = X[numeric_columns].astype('float64')
    
    return X, y

# Ana başlık
st.title("Model Evaluation")

# Model ve veriyi yükle
model, explainer = load_model_and_explainer()
X, y = load_and_prepare_data()

if model is not None and explainer is not None:
    # Tahminler
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred)
    
    conf_matrix_df = pd.DataFrame(cm,
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
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        "Not Churn": {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, pos_label=0),
            "Recall": recall_score(y, y_pred, pos_label=0),
            "F1 Score": f1_score(y, y_pred, pos_label=0),
            "AUC Score": roc_auc_score(y, y_pred_proba[:, 1])
        },
        "Churn": {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, pos_label=1),
            "Recall": recall_score(y, y_pred, pos_label=1),
            "F1 Score": f1_score(y, y_pred, pos_label=1),
            "AUC Score": roc_auc_score(y, y_pred_proba[:, 1])
        }
    }
    
    # Metrikleri göster
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

    # Feature Importance Analysis
    st.subheader("Feature Importance Analysis")
    
    # Model ve veriyi yükle
    df = pd.read_csv("customer_churn_records.csv")
    X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Complain'], axis=1)
    y = df['Exited']

    # Kategorik değişkenleri dönüştür
    categorical_columns = ['Geography', 'Gender', 'Card Type']
    for column in categorical_columns:
        if column == 'Geography':
            X[column] = X[column].map({'France': 0, 'Germany': 1, 'Spain': 2})
        elif column == 'Gender':
            X[column] = X[column].map({'Female': 0, 'Male': 1})
        elif column == 'Card Type':
            X[column] = X[column].map({'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3})

    # Model oluştur ve eğit
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Feature importance'ı direkt modelden al
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=True)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.barh(importances['feature'], importances['importance'], color='#2E86C1')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()

    # Streamlit'te göster
    st.pyplot(fig)

    # Tek bir örnek için tahmin ve olasılıkları göster
    sample_idx = st.slider('Select customer index', 0, len(X)-1, 0)
    X_sample = X.iloc[sample_idx:sample_idx+1]
    pred_proba = model.predict_proba(X_sample)[0]

    st.write(f"\nSample Customer (Index {sample_idx}) Analysis:")
    st.write(f"Probability of Churn: {pred_proba[1]:.3f}")
    
    # Müşteri özelliklerini tablo olarak göster
    st.write("\nFeature Values:")
    feature_df = pd.DataFrame(X_sample.iloc[0]).reset_index()
    feature_df.columns = ['Feature', 'Value']
    st.table(feature_df)

    # ROC Curve ve Threshold Optimization
    st.subheader("ROC Curve & Threshold Optimization")
    
    # Model ve veriyi hazırla
    from sklearn.model_selection import train_test_split
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model eğitimi
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Test seti üzerinde tahminler
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC eğrisi hesaplama
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # ROC grafiği
    fig_roc = plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, color='#2E86C1', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_roc)
    
    # Threshold Optimization
    st.subheader("Threshold Optimization")
    
    # Precision ve Recall hesaplama
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    
    # F1 score hesaplama
    f1_scores = []
    for pr, re in zip(precision[:-1], recall[:-1]):
        if pr + re == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * (pr * re) / (pr + re))
    
    # Threshold vs Metrics grafiği
    fig_thresh = plt.figure(figsize=(12, 6))
    plt.plot(thresholds_pr, precision[:-1], '#2E86C1', label='Precision', linewidth=2)
    plt.plot(thresholds_pr, recall[:-1], '#ffbd59', label='Recall', linewidth=2)
    plt.plot(thresholds_pr, f1_scores, '#28B463', label='F1 Score', linewidth=2)
    
    # Optimal threshold'u bul
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx]
    
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_thresh)
    
    # Optimal threshold sonuçları
    st.write("\nOptimal Threshold Analysis:")
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    from sklearn.metrics import classification_report
    optimal_report = classification_report(y_test, y_pred_optimal, output_dict=True)
    
    # Sonuçları DataFrame'e çevir
    df_optimal = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Not Churn': [optimal_report['0']['precision'], 
                     optimal_report['0']['recall'], 
                     optimal_report['0']['f1-score']],
        'Churn': [optimal_report['1']['precision'], 
                 optimal_report['1']['recall'], 
                 optimal_report['1']['f1-score']]
    })
    
    # Metrikleri göster
    st.write("Performance Metrics with Optimal Threshold:")
    st.table(df_optimal.style.format({
        'Not Churn': '{:.3f}',
        'Churn': '{:.3f}'
    }))
    
    # İnteraktif threshold seçimi
    st.subheader("Interactive Threshold Selection")
    custom_threshold = st.slider(
        "Select Custom Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(optimal_threshold),
        step=0.01
    )
    
    # Seçilen threshold ile sonuçlar
    y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)
    custom_report = classification_report(y_test, y_pred_custom, output_dict=True)
    
    df_custom = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Not Churn': [custom_report['0']['precision'], 
                     custom_report['0']['recall'], 
                     custom_report['0']['f1-score']],
        'Churn': [custom_report['1']['precision'], 
                 custom_report['1']['recall'], 
                 custom_report['1']['f1-score']]
    })
    
    st.write(f"Performance Metrics with Custom Threshold ({custom_threshold:.2f}):")
    st.table(df_custom.style.format({
        'Not Churn': '{:.3f}',
        'Churn': '{:.3f}'
    }))

else:
    st.error("Could not load model or explainer. Please check if the files exist and are valid.")