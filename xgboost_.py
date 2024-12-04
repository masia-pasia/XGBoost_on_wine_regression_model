# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Wczytaj dane z pliku CSV
input_file = "wina.csv"  # Podaj nazwę pliku wejściowego
output_file = "wina_przetworzone.csv"  # Podaj nazwę pliku wyjściowego

# Wczytanie pliku do DataFrame
df = pd.read_csv(input_file)

# Usuń kolumny 0 i 2 oraz dodatkową kolumnę (6)
df_dropped = df.drop(df.columns[[0, 2, 6]], axis=1)

# ODRZUCIC WINNICE KTORYCH SREDNIA ILOSCI OPINII JEST BARDZO MALA
# PRZETESTOWAC LAS LOSOWY

# Zastosuj one-hot encoding dla kolumn kategorialnych
df_encoded = pd.get_dummies(df_dropped, drop_first=True)

# Zapisz przetworzone dane do nowego pliku CSV
df_encoded.to_csv(output_file, index=False)
print(df_encoded.columns.size)

print(f"Przetworzony plik zapisano jako: {output_file}")

# Usunięcie rekordów z wartością Price > 1000
df_encoded = df_encoded.loc[~(df_encoded['Price'] > 1000)].copy()
df_encoded = df_encoded.loc[~(df_encoded['num_review'] > 1000)].copy()

# Wyodrębnij zmienną docelową i cechy
y = df_encoded['Rating']
df_encoded.drop(columns=['Rating'], inplace=True)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.25, random_state=42)

print(y_train.size)
# Inicjalizacja i trenowanie modelu XGBoost
# teoretycznie czym wieksza glebokosc i czym mniejsza min ilosc probek w lisciu tym wieksza eksploatacja (lepsze dopasowanie do danych)
# czym wiecej estymatorow tym wieksza eksploatacja - dostrajamy kolejne estymatory do danych
regr_1 = XGBRegressor(max_depth=6, min_child_weight=3, n_estimators=20, random_state=42)
regr_2 = XGBRegressor(max_depth=10, min_child_weight=2, n_estimators=100, random_state=42)

# Trenowanie modeli
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Funkcja do obliczenia RMSE i MAE
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Ocena modeli
evaluate_model(y_test, y_1, "Model 1 (max_depth=1000, min_child_weight=3, n_estimators=20)")
evaluate_model(y_test, y_2, "Model 2 (max_depth=1000, min_child_weight=2, n_estimators=100)")


# Wizualizacja wyników
plt.figure()

# Rozrzut punktów
plt.scatter(y_test, y_1, color="cornflowerblue", label="max_depth=5, min_child_weight=10", linewidth=2)
plt.scatter(y_test, y_2, color="yellowgreen", label="max_depth=10, min_child_weight=3", linewidth=2)

# Ustawienia osi logarytmicznych
plt.xscale('log')
plt.yscale('log')

# Etykiety i tytuł
plt.xlabel("data (log scale)")
plt.ylabel("target (log scale)")
plt.title("XGBoost Regression (Logarithmic Scale)")

# Legenda i wyświetlenie
plt.legend()
plt.show()
