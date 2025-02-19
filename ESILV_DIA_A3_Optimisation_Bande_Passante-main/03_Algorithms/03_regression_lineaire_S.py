import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("01_Data_to_use/03_04_05_06_Dataframe_reglin.csv")
print(df)

feature_simple = "Total des voies d'accès de communication"
target = "Total des recettes des télécommunications USD"

X = df[[feature_simple]]  
y = df[target]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression linéaire simple
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Évaluation du modèle
r2 = round(r2_score(y_test, y_pred), 4)
mse = round(mean_squared_error(y_test, y_pred), 4)
rmse = round(np.sqrt(mse), 4)

rmse_normalized = round(rmse / np.mean(y_test), 4)

print("Coefficient de détermination (R²) :", r2)
print("Mean Squared Error (MSE) :", mse)
print("Root Mean Squared Error (RMSE) :", rmse)
print("RMSE normalisé :", rmse_normalized)

# Visualisation sur les données de test
plt.figure(figsize=(10, 6))

X_test_sorted = X_test.sort_values(by=feature_simple)
y_pred_sorted = model.predict(X_test_sorted)

plt.scatter(X_test, y_test, color='blue', label='Valeurs réelles', alpha=0.6)
plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Régression linéaire')
plt.xlabel(feature_simple)
plt.ylabel(target)
plt.title("Régression linéaire simple")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

# Prédiction sur un autre DataFrame (dftest)
dftest = pd.read_csv('01_Data_to_use/04_Dataframe_reglin_1.csv')
dftest = dftest[~dftest['Pays'].isin(['OCDE - Total', 'États-Unis', 'Chili', "Colombie"])]

X_test_new = dftest[[feature_simple]] 
y_test_real = dftest[target]

y_pred_new = model.predict(X_test_new)

# Évaluation sur dftest
r2_new = round(r2_score(y_test_real, y_pred_new), 4)
mse_new = round(mean_squared_error(y_test_real, y_pred_new), 4)
rmse_new = round(np.sqrt(mse_new), 4)

print("\nÉvaluation sur le DataFrame de test (dftest) :")
print("Coefficient de détermination (R²) :", r2_new)
print("Mean Squared Error (MSE) :", mse_new)
print("Root Mean Squared Error (RMSE) :", rmse_new)

# Visualisation des points et de la droite de régression pour dftest
plt.figure(figsize=(10, 6))

X_test_new_sorted = X_test_new.sort_values(by=feature_simple)
y_pred_new_sorted = model.predict(X_test_new_sorted)

plt.scatter(X_test_new, y_test_real, color='blue', label='Valeurs réelles (dftest)', alpha=0.6)
plt.plot(X_test_new_sorted, y_pred_new_sorted, color='red', label='Régression linéaire (dftest)')
plt.xlabel(feature_simple)
plt.ylabel(target)
plt.title("Régression linéaire simple pour le DataFrame dftest")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

