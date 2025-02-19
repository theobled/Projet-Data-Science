# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Étape 1 : Charger les données depuis le fichier CSV
# df = pd.read_csv("export_dataframe_vb_groupby.csv")

# # Étape 2 : Nettoyage des données
# # Identifier la colonne contenant les noms des pays
# if "Pays" not in df.columns:
#     raise ValueError("La colonne 'Pays' est absente des données. Vérifiez le fichier CSV.")

# # Extraire les noms des pays pour référence
# pays = df["Pays"]  # Garder une référence pour les résultats
# df = df.drop(columns=["Pays"])  # Supprimer la colonne avant l'analyse

# # Remplacer les valeurs manquantes par la moyenne de chaque colonne
# df.fillna(df.mean(), inplace=True)

# # Étape 3 : Normalisation des données
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df)

# # Étape 4 : Trouver le nombre optimal de clusters avec la méthode du coude
# inertias = []
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(df_scaled)
#     inertias.append(kmeans.inertia_)

# # Visualisation de la méthode du coude
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 10), inertias, marker='o')
# plt.xlabel('Nombre de clusters')
# plt.ylabel('Inertie')
# plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
# plt.show()

# # Étape 5 : Appliquer K-means avec un nombre de clusters optimal (ex : 3)
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(df_scaled)

# # Ajout des clusters et noms des pays dans le DataFrame d'origine
# df['Cluster'] = clusters
# df['Pays'] = pays

# # Étape 6 : Visualisation des clusters
# sns.scatterplot(
#     data=df,
#     x="Abonnements_prépayés",  # Remplacez par les colonnes pertinentes
#     y="Recettes_totales_USD",  # Remplacez par les colonnes pertinentes
#     hue="Cluster",
#     palette="viridis",
#     s=100
# )
# plt.title('Clustering K-means des pays')
# plt.xlabel('Abonnements prépayés')
# plt.ylabel('Recettes totales (USD)')
# plt.legend(title='Cluster')
# plt.show()

# # Affichage des données avec clusters
# print(df[['Pays', 'Cluster']])


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # Charger le dataframe
# df = pd.read_csv("export_dataframe_vb_groupby.csv")

# # Exclure la colonne 'Pays' pour éviter les problèmes avec les données non numériques
# df_numeric = df.drop(columns=['Pays'])

# # Standardiser les données
# scaler = StandardScaler()
# df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# # Afficher une heatmap de corrélation pour les données standardisées
# plt.figure(figsize=(12, 8))
# heatmap = sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Heatmap des Corrélations des Données Standardisées")
# plt.show()






import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Charger le dataframe (assurez-vous que le chemin est correct)
df = pd.read_csv("Dataframe_v3.csv")

# Supprimer les lignes où le pays est "États-Unis" ou "OCDE"
df = df[~df['Pays'].isin(['États-Unis', 'OCDE - Total'])]

# Sélectionner uniquement les colonnes pertinentes
df_selected = df[['Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD', 
                  'Total des abonnements au téléphone cellulaire mobile']]

# Supprimer les lignes avec des valeurs manquantes dans les colonnes pertinentes
df_selected_clean = df_selected.dropna()

# Standardiser les données
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected_clean)

# Appliquer KMeans (par exemple, avec 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df_selected_clean['Cluster'] = kmeans.fit_predict(df_scaled)

# Associer les clusters au DataFrame original en fonction de l'index
df['Cluster'] = None
df.loc[df_selected_clean.index, 'Cluster'] = df_selected_clean['Cluster']

# Visualiser les clusters (optionnel, si vous souhaitez voir le résultat)
plt.scatter(df_selected_clean['Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD'],
            df_selected_clean['Total des abonnements au téléphone cellulaire mobile'], c=df_selected_clean['Cluster'], cmap='viridis')
plt.xlabel('Investissements totaux dans les télécommunications (USD)')
plt.ylabel('Total des abonnements au téléphone cellulaire mobile')
plt.title('Clustering KMeans')
plt.colorbar(label='Cluster')

# Ajouter des annotations pour les pays
for i, country in enumerate(df.loc[df_selected_clean.index, 'Pays']):
    plt.annotate(country, (df_selected_clean.iloc[i, 0], df_selected_clean.iloc[i, 1]),
                 fontsize=8, alpha=0.7, ha='right')
    
plt.show()

# Afficher les résultats des clusters
print(df[['Pays', 'Cluster']])

