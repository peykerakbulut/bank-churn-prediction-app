import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import pickle
import os

# Klasör yollarını oluştur
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# models klasörünü oluştur
os.makedirs(MODELS_DIR, exist_ok=True)

# Veriyi yükle
df = pd.read_csv(os.path.join(DATA_DIR, "customer_churn_records.csv"))

# Feature isimlerini düzenle
feature_names = {
    'CreditScore': 'Credit Score',
    'Age': 'Age',
    'Tenure': 'Tenure',
    'Balance': 'Balance',
    'NumOfProducts': 'Number of Products',
    'HasCrCard': 'Has Credit Card',
    'IsActiveMember': 'Is Active Member',
    'EstimatedSalary': 'Estimated Salary',
    'Satisfaction Score': 'Satisfaction Score',
    'Point Earned': 'Points Earned',
    'Geography': 'Country',
    'Gender': 'Gender',
    'Card Type': 'Card Type'
}

# Veriyi hazırla - Complain'i çıkar
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Complain'], axis=1)
# Feature isimlerini değiştir
X = X.rename(columns=feature_names)
y = df['Exited']

# Kategorik değişkenleri dönüştür
categorical_columns = ['Country', 'Gender', 'Card Type']
label_encoders = {}

# Label Encoding uygula
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Sayısal değerleri float64'e dönüştür
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_columns] = X[numeric_columns].astype('float64')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# SHAP explainer oluştur
X_sample = X_train.iloc[:1000]  # İlk 1000 örnek
explainer = shap.TreeExplainer(model)

# Test et
try:
    shap_values = explainer.shap_values(X_sample)
    print("SHAP values calculated successfully!")
except Exception as e:
    print("Error calculating SHAP values:", e)

# Dosyaları kaydet
model_path = os.path.join(MODELS_DIR, 'model.pkl')
explainer_path = os.path.join(MODELS_DIR, 'explainer.pkl')
feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
label_encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')

with open(model_path, 'wb') as file:
    pickle.dump(model, file)

with open(explainer_path, 'wb') as file:
    pickle.dump(explainer, file)

with open(feature_names_path, 'wb') as file:
    pickle.dump(feature_names, file)

with open(label_encoders_path, 'wb') as file:
    pickle.dump(label_encoders, file)

print("Model, explainer, encoders and feature names saved successfully!")

# Model performansını test et
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print("\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 