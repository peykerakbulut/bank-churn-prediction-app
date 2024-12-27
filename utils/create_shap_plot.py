import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit başlığı
st.title("Feature Importance Analysis")

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
st.write("\nFeature Values:")
for col, val in X_sample.iloc[0].items():
    st.write(f"{col}: {val}") 