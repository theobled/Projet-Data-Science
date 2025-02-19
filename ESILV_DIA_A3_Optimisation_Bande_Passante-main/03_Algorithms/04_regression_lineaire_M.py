import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("01_Data_to_use/03_04_05_06_Dataframe_reglin.csv")
print(df)

features = [
    "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD",
    "Total des lignes d'accès téléphoniques",
    "Total des voies d'accès de communication"
]
target = "Total des recettes des télécommunications USD"

X = df[features]
y = df[target]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur un autre DataFrame (dftest)
dftest = pd.read_csv('01_Data_to_use/04_Dataframe_reglin_1.csv')
dftest = dftest[~dftest['Pays'].isin(['OCDE - Total', 'États-Unis', 'Chili', 'Colombie'])]

X_test_new = dftest[features]
y_test_real = dftest[target]
y_pred_new = model.predict(X_test_new)

# Calcul du R² pour le nouveau dataset
r2_new = r2_score(y_test_real, y_pred_new)
print(f"Coefficient de détermination (R²) pour les prédictions sur le nouveau dataset : {r2_new:.4f}")

countries = dftest['Pays']

# Comparaison des valeurs réelles et prédites pour chaque pays
plt.figure(figsize=(12, 6))
plt.plot(countries, y_test_real, label="Valeurs réelles", marker='o', alpha=0.7, color='blue')
plt.plot(countries, y_pred_new, label="Valeurs prédites", marker='x', alpha=0.7, linestyle='--', color='orange')
plt.xlabel("Pays")
plt.ylabel("Total des recettes des télécommunications USD")
plt.title(f"Comparaison des valeurs réelles et prédites pour chaque pays (R² = {r2_new:.4f})")
plt.legend()
plt.grid(alpha=0.5)
plt.xticks(rotation=90)  
plt.tight_layout()
plt.show()