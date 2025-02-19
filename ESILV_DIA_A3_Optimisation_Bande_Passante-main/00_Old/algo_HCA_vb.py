import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Appliquer l'algorithme HCA (regroupement hiérarchique)
hca = AgglomerativeClustering(n_clusters=3)  # Ici, on suppose 3 clusters, vous pouvez ajuster selon vos besoins
df_selected_clean['Cluster'] = hca.fit_predict(df_scaled)

# Associer les clusters au DataFrame original en fonction de l'index
df['Cluster'] = None
df.loc[df_selected_clean.index, 'Cluster'] = df_selected_clean['Cluster']

# Visualisation avec un dendrogramme
plt.figure(figsize=(10, 6))
linked = linkage(df_scaled, method='ward')  # 'ward' minimise la variance dans les clusters
dendrogram(linked, labels=df_selected_clean.index, orientation='top')
plt.title('Dendrogramme HCA')
plt.xlabel('Pays')
plt.ylabel('Distance')
plt.show()

# Visualiser les clusters avec un scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_selected_clean['Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD'],
                      df_selected_clean['Total des abonnements au téléphone cellulaire mobile'], 
                      c=df_selected_clean['Cluster'], cmap='viridis')

# Ajouter une légende avec des couleurs correspondant aux clusters
plt.xlabel('Investissements totaux dans les télécommunications (USD)')
plt.ylabel('Total des abonnements au téléphone cellulaire mobile')
plt.title('Clustering Hiérarchique')
plt.colorbar(scatter, label='Cluster')

# Ajouter des annotations pour les pays
for i, country in enumerate(df.loc[df_selected_clean.index, 'Pays']):
    plt.annotate(country, (df_selected_clean.iloc[i, 0], df_selected_clean.iloc[i, 1]),
                 fontsize=8, alpha=0.7, ha='right')

plt.show()

# Afficher les résultats des clusters
print(df[['Pays', 'Cluster']])
