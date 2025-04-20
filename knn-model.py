# 1. Gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 2. Veri setinin yüklenmesi ve incelenmesi
# Iris veri setini yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi DataFrame'e dönüştürme
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Veriyi inceleme
print("Veri setinin ilk 5 satırı:")
print(df.head())
print("\nVeri setinin istatistikleri:")
print(df.describe())
print("\nHedef değişken dağılımı:")
print(df['target'].value_counts())

# 3. Verinin eğitim ve test kümelerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# 4. Verinin Standardizasyon
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nÖlçeklendirilmiş eğitim verisinin ilk 5 satırı:")
print(X_train[:5])

# 5. KNN modelinin oluşturulması ve eğitilmesi
k = int(np.sqrt(len(X_train)))
print(f"\nBaşlangıç k değeri: {k}")

# Modeli oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 6. Tahmin yapma
y_pred = knn.predict(X_test)
print("\nModel Değerlendirme Sonuçları:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy Score (k={k}): {accuracy_score(y_test, y_pred):.4f}")

# 7. Optimum k değerinin belirlenmesi
error_rate = []
k_values = range(1, 40)

for i in k_values:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
# Hata oranını görselleştirme
plt.figure(figsize=(10,6))
plt.plot(k_values, error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

# En düşük hata oranına sahip k değeri
optimal_k = error_rate.index(min(error_rate)) + 1
print(f"\nOptimal k değeri: {optimal_k}")

# Modeli optimal k ile yeniden eğitme
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
y_pred_optimal = knn_optimal.predict(X_test)

# Optimal modelin performansını değerlendirme
print("\nOptimal Model Değerlendirme Sonuçları:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_optimal))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal))
print(f"\nOptimal Model Accuracy (k={optimal_k}): {accuracy_score(y_test, y_pred_optimal):.4f}")

# Sonuçların karşılaştırılması
improvement = (accuracy_score(y_test, y_pred_optimal) - accuracy_score(y_test, y_pred))
print(f"\nDoğrulukta iyileşme: {improvement:.4f} ({improvement*100:.2f}%)")