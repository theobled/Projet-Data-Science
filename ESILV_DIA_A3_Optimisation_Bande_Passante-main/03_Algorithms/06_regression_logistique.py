from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèle de régression logistique
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Coefficients du modèle
coefficients = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": log_reg.coef_[0]
})
print("=== Coefficients de la Régression Logistique ===")
print(coefficients)

# Prédictions et évaluation sur les données de test
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Probabilités pour la classe "élevé"

print("Régression Logistique")
print("Rapport de classification (jeu de test) :")
print(classification_report(y_test, y_pred, target_names=["faible", "élevé"]))

# Matrice de confusion pour le jeu de test
conf_matrix_test = confusion_matrix(y_test, y_pred)
accuracy_test = accuracy_score(y_test, y_pred)

print("Matrice de confusion (jeu de test) :")
print(conf_matrix_test)
print(f"Précision (jeu de test) : {accuracy_test:.2f}")

# Matrice de confusion graphique pour le jeu de test
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=["faible", "élevé"], yticklabels=["faible", "élevé"])
plt.xlabel("Prédictions")
plt.ylabel("Vrai")
plt.title("Matrice de Confusion - Jeu de Test")
plt.show()

# Courbe ROC pour le jeu de test
fpr, tpr, thresholds = roc_curve(y_test.cat.codes, y_pred_proba)
auc_score = roc_auc_score(y_test.cat.codes, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Courbe ROC (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="No Skill")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbe ROC - Jeu de Test")
plt.legend()
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

# Normalisation des nouvelles données
X_new_scaled = scaler.transform(X_new)

# Prédictions sur le nouveau dataframe
new_df["Categorie_Predite"] = log_reg.predict(X_new_scaled)

# Évaluation sur le nouveau dataframe
conf_matrix_new = confusion_matrix(new_df["Categorie_Abonnements"], new_df["Categorie_Predite"])
accuracy_new = accuracy_score(new_df["Categorie_Abonnements"], new_df["Categorie_Predite"])

print("=== Évaluation sur le Nouveau DataFrame ===")
print("Matrice de confusion (nouveau dataframe) :")
print(conf_matrix_new)
print(f"Précision (nouveau dataframe) : {accuracy_new:.2f}")

# Matrice de confusion graphique pour le nouveau dataframe
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_new, annot=True, fmt="d", cmap="Blues", xticklabels=["faible", "élevé"], yticklabels=["faible", "élevé"])
plt.xlabel("Prédictions")
plt.ylabel("Vrai")
plt.title("Matrice de Confusion - Nouveau DataFrame")
plt.show()