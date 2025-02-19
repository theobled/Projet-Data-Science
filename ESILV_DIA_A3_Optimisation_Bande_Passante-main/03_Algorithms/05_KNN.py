from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("01_Data_to_use/03_04_05_06_Dataframe_reglin.csv")
df.set_index("Pays", inplace=True)

print(df.isnull().sum())
df = df.fillna(df.mean())

df["Categorie_Abonnements"] = pd.qcut(
    df["Total des recettes des télécommunications USD"],
    q=2,
    labels=["faible", "élevé"]
)

X = df[["Abonnements au téléphone cellulaire mobile utilisant des cartes prépayés",
        "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD",
        "Total des abonnements au téléphone cellulaire mobile",
        "Total des lignes d'accès téléphoniques",
        "Total des voies d'accès de communication"]]
y = df["Categorie_Abonnements"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Recherche du meilleur k avec validation croisée
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_range[np.argmax(cv_scores)]
print(f"Le k optimal est : {optimal_k}")

# Visualisation de la précision en fonction de k
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o')
plt.title('Validation croisée pour déterminer k')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Précision moyenne')
plt.xticks(k_range)
plt.grid()
plt.show()

# Entraînement du modèle avec le k optimal
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Evaluation
y_pred_test = knn.predict(X_test)

# Matrice de confusion pour le jeu de test
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("=== Évaluation sur le Jeu de Test ===")
print("Matrice de confusion (jeu de test) :")
print(conf_matrix_test)
print(f"Précision (jeu de test) : {accuracy_test:.2f}")

# Matrice de confusion graphique pour le jeu de test
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(conf_matrix_test, display_labels=["faible", "élevé"]).plot(cmap="Blues")
plt.title("Matrice de Confusion - Jeu de Test")
plt.show()

# Prédiction sur le nouveau dataframe 
new_df = pd.read_csv("01_Data_to_use/04_Dataframe_reglin_1.csv")
new_df.set_index("Pays", inplace=True)

# Vérification des valeurs manquantes et remplissage
print(new_df.isnull().sum())
new_df = new_df.fillna(new_df.mean())

# Création de la colonne cible pour le nouveau dataframe
new_df["Categorie_Abonnements"] = pd.qcut(
    new_df["Total des recettes des télécommunications USD"],
    q=2,
    labels=["faible", "élevé"]
)

# Sélection des mêmes colonnes explicatives
X_new = new_df[["Abonnements au téléphone cellulaire mobile utilisant des cartes prépayés",
                "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD",
                "Total des abonnements au téléphone cellulaire mobile",
                "Total des lignes d'accès téléphoniques",
                "Total des voies d'accès de communication"]]


X_new_scaled = scaler.transform(X_new)
new_df["Categorie_Predite"] = knn.predict(X_new_scaled)

# Matrice de confusion pour le nouveau dataframe
conf_matrix_new = confusion_matrix(new_df["Categorie_Abonnements"], new_df["Categorie_Predite"])
accuracy_new = accuracy_score(new_df["Categorie_Abonnements"], new_df["Categorie_Predite"])

print("=== Évaluation sur le Nouveau DataFrame ===")
print("Matrice de confusion (nouveau dataframe) :")
print(conf_matrix_new)
print(f"Précision (nouveau dataframe) : {accuracy_new:.2f}")

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(conf_matrix_new, display_labels=["faible", "élevé"]).plot(cmap="Blues")
plt.title("Matrice de Confusion - Nouveau DataFrame")
plt.show()
