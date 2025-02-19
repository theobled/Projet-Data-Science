# Description: Implémentation de l'algorithme KNN pour la prédiction de la catégorie d'abonnements prépayés
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

# Charger le dataframe (assurez-vous que le chemin est correct)
df = pd.read_csv("Dataframe_v4(Mehdi).csv")

# S'assurer que la colonne 'Pays' est utilisée comme index
df.set_index("Pays", inplace=True)

print(df.isnull().sum())  # Affiche le nombre de NaN par colonne
df = df.fillna(df.mean()) # Compléter les Nan par la moyenne


# Étape 1 : Discrétiser la cible
# Créons trois catégories pour "Abonnements_Prepayes" : faible, moyen, élevé (quantiles)
df["Categorie_Abonnements"] = pd.qcut(
    df["Total des recettes des télécommunications USD"], 
    q=2, 
    labels=["faible",  "élevé"]
)

# Étape 2 : Préparer les variables explicatives et la cible
X = df[["Abonnements au téléphone cellulaire mobile utilisant des cartes prépayés",
        "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD", 
        "Total des abonnements au téléphone cellulaire mobile",
        "Total des lignes d'accès téléphoniques", 
        "Total des voies d'accès de communication"]]
y = df["Categorie_Abonnements"]

df.to_csv('30122024.csv', encoding="utf-8") 


# Étape 2 : Préparer les variables explicatives et la cible
# X = data_cleaned[["Investissements_USD", "Total_Abonnements_Mobiles",
#                   "Lignes_Acces_Telephoniques", "Recettes_Telecommunications_USD",
#                   "Voies_Acces_Communication"]]

# Étape 3 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 4 : Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Étape 5 : Mise en place du modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Étape 6 : Prédictions et évaluation
y_pred = knn.predict(X_test)

print("Classes présentes dans y_test :", np.unique(y_test))
print("Classes présentes dans y_pred :", np.unique(y_pred))


# Rapport de classification
report = classification_report(y_test, y_pred, target_names=["faible", "élevé"])

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Précision du modèle
accuracy = accuracy_score(y_test, y_pred)

# Affichage des résultats
print("Rapport de classification :")
print(report)

print("Matrice de confusion :")
print(conf_matrix)

print(f"Précision du modèle : {accuracy:.2f}")
