import tensorflow as tf
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor

# ----------- 1. Przygotowywanie danych

# 1.1 pobranie danych z UCI
automobile = fetch_ucirepo(id=10)

# 1.2 przekształcenie danych na DataFrame
X = automobile.data.features

# 1.3 wyswietlenie informacji
# print("Pierwsze 5 wierszy danych:")
# print(X.head())
# print("\nInformacje o danych:")
# print(X.info())
# print("\nLiczba brakujących wartości w każdej kolumnie:")
# print(X.isnull().sum())


# ----------- 2. Imputacja brakujących danych

# 2.1 dzielenie danych i imputacja
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
numerical_imputer = SimpleImputer(strategy='mean')
X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

categorical_columns = X.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
X = pd.get_dummies(X, columns=categorical_columns)

# 2.2 wyswietlenie przygotowanych danych
# print("Przygotowane dane:")
# print(X.head())

# ----------- 3. Przygotowanie i trenowanie modelu regresji kwantylowej

# 3.1 usunięcie szukanej cechy
y = X['price']
X = X.drop('price', axis=1)

# 3.2 podział danych na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.3 budowa modelu regresji kwantylowej (np. dla mediany, kwantyl 0.5)
model_quantile = QuantileRegressor(quantile=0.5, alpha=0.0)  # alpha=0.0 oznacza brak regularizacji

# 3.4 kompilacja i trenowanie modelu
model_quantile.fit(X_train, y_train)

# 3.5 ewaluacja modelu na zbiorze testowym
y_pred_quantile = model_quantile.predict(X_test)
test_loss_quantile = mean_squared_error(y_test, y_pred_quantile)

# ----------- 4. Wyswietlenie wykresu "Rzeczywiste vs Przewidywane"

# 4.1 tworzenie wykresu dla regresji kwantylowej
indices = np.arange(len(y_test))

# 4.2 tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.bar(indices - 0.2, y_test, width=0.4, color='red', label='Rzeczywista cena')
plt.bar(indices + 0.2, y_pred_quantile.flatten(), width=0.4, color='blue', label='Przewidywana cena (Kwantyl 0.5)')
plt.xlabel('Samochody')
plt.ylabel('Cena')
plt.title('Rzeczywiste vs Przewidywane ceny samochodów (Regresja Kwantylowa)')
plt.legend()
plt.xticks(indices, [f"Auto {i+1}" for i in indices], rotation=90)
plt.tight_layout()
# plt.show()

# ----------- 5. Ocena modelu i analiza wyników

# 5.1 obliczanie miar jakości modelu
mae_quantile = mean_absolute_error(y_test, y_pred_quantile)
mse_quantile = mean_squared_error(y_test, y_pred_quantile)
rmse_quantile = np.sqrt(mse_quantile)
r2_quantile = r2_score(y_test, y_pred_quantile)

# 5.2 przygotowanie wyników w postaci tabeli
metrics_quantile_extended = {
    "Metric": ["MAE", "MSE", "RMSE", "R²", "Accuracy", "Loss", "Histogram błędów", "Porównanie wykresu"],
    "Value": [
        mae_quantile,         # MAE
        mse_quantile,         # MSE
        rmse_quantile,        # RMSE
        r2_quantile,          # R²
        r2_quantile,          # Accuracy (możemy przyjąć R² jako przybliżenie dla regresji)
        test_loss_quantile,
        "Zobacz histogram",   # Link do wykresu histogramu błędów
        "Zobacz wykres"       # Link do wykresu rzeczywiste vs przewidywane
    ]
}

# 5.3 tworzenie tabeli i wyswietlanie
results_quantile_extended_df = pd.DataFrame(metrics_quantile_extended)
print(results_quantile_extended_df)

# 5.5 wizualizacja rozkladu bledow
errors_quantile = y_pred_quantile.flatten() - y_test
plt.figure(figsize=(10, 6))
plt.hist(errors_quantile, bins=30, color='gray', edgecolor='black')
plt.title('Histogram Błędów (Rzeczywiste - Przewidywane) - Regresja Kwantylowa')
plt.xlabel('Błąd (Rzeczywista - Przewidywana cena)')
plt.ylabel('Liczba samochodów')
# plt.show()
