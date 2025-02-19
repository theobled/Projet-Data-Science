# Description: Implémentation de la régression logistique pour la prédiction de la catégorie d'abonnements prépayés

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataframe (assurez-vous que le chemin est correct)
df = pd.read_csv("Dataframe_v4(Mehdi).csv")

# S'assurer que la colonne 'Pays' est utilisée comme index
df.set_index("Pays", inplace=True)

# Affichage des valeurs manquantes et traitement
print(df.isnull().sum())  # Affiche le nombre de NaN par colonne
df = df.fillna(df.mean())  # Compléter les NaN par la moyenne

# Étape 1 : Discrétiser la cible
df["Categorie_Abonnements"] = pd.qcut(
    df["Total des recettes des télécommunications USD"], 
    q=2, 
    labels=["faible", "élevé"]
)

# Étape 2 : Préparer les variables explicatives et la cible
X = df[["Abonnements au téléphone cellulaire mobile utilisant des cartes prépayés",
        "Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD", 
        "Total des abonnements au téléphone cellulaire mobile",
        "Total des lignes d'accès téléphoniques", 
        "Total des voies d'accès de communication"]]
y = df["Categorie_Abonnements"]

# Étape 3 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 4 : Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Étape 5 : Mise en place du modèle de régression logistique
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Coefficients du modèle 
coefficients = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": log_reg.coef_[0]
})
print("=== Coefficients de la Régression Logistique ===")
print(coefficients)

# Étape 6 : Prédictions et évaluation
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Probabilités pour la classe "élevé"

# Évaluation du modèle
print("=== Régression Logistique ===")
print("Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=["faible", "élevé"]))

print("Matrice de confusion :")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print(f"Précision du modèle : {accuracy_score(y_test, y_pred):.2f}")

# Affichage graphique de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["faible", "élevé"], yticklabels=["faible", "élevé"])
plt.xlabel("Prédictions")
plt.ylabel("Vrai")
plt.title("Matrice de Confusion")
plt.show()

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test.cat.codes, y_pred_proba)
auc_score = roc_auc_score(y_test.cat.codes, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="No Skill")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbe ROC")
plt.legend()
plt.show()