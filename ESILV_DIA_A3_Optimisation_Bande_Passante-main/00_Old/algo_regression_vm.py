import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Chargement des données
df = pd.read_csv("Dataframe_v4(Mehdi).csv")

# Liste des variables explicatives (features) et de la cible (target)
features = [
    "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD",
    "Total des lignes d'accès téléphoniques",
    "Total des voies d'accès de communication"
]
target = "Total des recettes des télécommunications USD"

# Suppression des lignes avec des valeurs manquantes
df = df.dropna(subset=features + [target])

# Variables explicatives (X) et cible (y)
X = df[features]
y = df[target]

# Centrage-réduction des variables explicatives
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ajout des colonnes centrées-réduites au DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
df_scaled = pd.concat([df.drop(columns=features), X_scaled_df], axis=1)

# Vérification : Moyenne et écart-type des nouvelles colonnes
print("Moyenne après centrage-réduction :\n", X_scaled_df.mean(axis=0))
print("Écart-type après centrage-réduction :\n", X_scaled_df.std(axis=0))

# Régression linéaire avec données centrées-réduites
model = LinearRegression()
model.fit(X_scaled_df, y)

# Ajout des prédictions dans le DataFrame centré-réduit
df_scaled['Recettes_prédites'] = model.predict(X_scaled_df)

# Sélection d'un pays pour la visualisation
pays_cible = "Australie"  # Remplacez par le pays de votre choix
df_pays_scaled = df_scaled[df_scaled['Pays'] == pays_cible]

# Graphique 1 : Comparaison des recettes réelles vs. prédites pour un pays donné
plt.figure(figsize=(10, 6))
plt.plot(
    df_pays_scaled['TIME_PERIOD'], 
    df_pays_scaled[target], 
    label="Recettes réelles", 
    marker='o'
)
plt.plot(
    df_pays_scaled['TIME_PERIOD'], 
    df_pays_scaled['Recettes_prédites'], 
    label="Recettes prédites (centrées)", 
    marker='x', 
    linestyle='--'
)
plt.xlabel("Année")
plt.ylabel("Recettes (USD)")
plt.title(f"Évolution des recettes réelles et prédites pour {pays_cible}")
plt.legend()
plt.grid()
plt.show()

# Graphique 2 : Toutes les années, scatter réel vs prédit
plt.figure(figsize=(10, 6))
plt.scatter(
    y, 
    df_scaled['Recettes_prédites'], 
    alpha=0.7, 
    label="Points de données"
)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', color='red', label="Idéal")
plt.xlabel("Recettes réelles")
plt.ylabel("Recettes prédites")
plt.title("Recettes réelles vs. prédites (tous les pays)")
plt.legend()
plt.grid()
plt.show()

# Calcul des scores globaux
r2 = r2_score(y, df_scaled['Recettes_prédites'])
rmse = np.sqrt(mean_squared_error(y, df_scaled['Recettes_prédites']))

print(f"Coefficient de détermination R^2 : {r2:.2f}")
print(f"Erreur quadratique moyenne (RMSE) : {rmse:.2f}")

# Exclure les valeurs réelles égales à zéro
df_non_zero = df[df[target] != 0]
mape = np.mean(np.abs((df_non_zero[target] - df_non_zero['Recettes_prédites']) / df_non_zero[target])) * 100
print(f"Erreur relative moyenne (MAPE) après correction : {mape:.2f}%")