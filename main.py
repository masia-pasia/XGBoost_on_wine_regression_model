# Import the necessary modules and libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Wczytaj dane z pliku CSV
input_file = "wina.csv"  # Podaj nazwę pliku wejściowego
output_file = "wina_przetworzone.csv"  # Podaj nazwę pliku wyjściowego

# Wczytanie pliku do DataFrame
df = pd.read_csv(input_file)

# Usuń kolumny 0 i 2
df_dropped = df.drop(df.columns[[0, 2, 6]], axis=1)

# Zastosuj one-hot encoding dla kolumn kategorialnych
df_encoded = pd.get_dummies(df_dropped, drop_first=True)

# Zapisz przetworzone dane do nowego pliku CSV
df_encoded.to_csv(output_file, index=False)
print(df_encoded.columns.size)

print(f"Przetworzony plik zapisano jako: {output_file}")

df_encoded = df_encoded.loc[~(df_encoded['Price'] > 1000)].copy()

# Create a random dataset
y = df_encoded['num_review']
df_encoded.drop(columns=['num_review'], inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.25, random_state=42)

print(y)
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=1000, min_samples_leaf=10)
regr_2 = DecisionTreeRegressor(max_depth=1000, min_samples_leaf=3)
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
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


plt.figure()

# Rozrzut punktów
# plt.scatter(df_encoded, y, s=20, edgecolor="black", c="darkorange", label="data")  # Jeżeli wymaga zakomentowania, pozostaw.
plt.scatter(y_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.scatter(y_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

# Ustawienia osi logarytmicznych
plt.xscale('log')
plt.yscale('log')

# Etykiety i tytuł
plt.xlabel("data (log scale)")
plt.ylabel("target (log scale)")
plt.title("Decision Tree Regression (Logarithmic Scale)")

# Legenda i wyświetlenie
plt.legend()
plt.show()