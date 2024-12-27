import pickle

# Model dosyasını yükle
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Modelin özellik isimlerini yazdır
print("Model feature names:")
print(model.feature_names_in_) 