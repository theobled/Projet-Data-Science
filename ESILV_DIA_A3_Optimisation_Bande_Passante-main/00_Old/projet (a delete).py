import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("OECD,DF_TEL,+all - 8000l.csv")

################################################################
# Suppression des colonnes contenants que des données vides
# Suppression des colonnes contenants que des données uniques
# Suppression des colonnes contenants des diminutifs
# Mettre la variable TIME_PERIOD en numérique pour faciliter de filtre
# Filtrer les lignes où TIME_PERIOD est supérieur à 1999

df = df.dropna(axis=1)
df = df.loc[:, df.nunique() > 1]
df = df.drop(["COU", "SER"], axis=1)
df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce').astype('Int64')
df = df[df['TIME_PERIOD'] > 1999]

################################################################
# Pivoter les colonnes "Série" pour les avoir en colonne
# Calcule du taux des valeurs vides dans chaque colonne
# Suppression des colonnes où la valeur est supérieur à 50% 
# Suppression des colonnes où la monnaie est en devise locale pour garder leurs équivalents en dollars
# Pas de correlation sur heatmap et c'est limite équivalent au total

df = df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE")
taux_vides = df.isnull().mean() * 100

df = df.drop(["Nombre d’abonnés à la télévision par câble", "Total Internet Protocol (IP) telephone subscriptions"], axis=1)
df = df.drop(["Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire)", "Total des recettes des télécommunications"], axis=1)
df = df.drop(["Total des voies d'accès de communication pour 100 habitants", "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], axis=1)

################################################################
# Option d'affichage du DataFrame 
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 15)     
df.to_csv('export_dataframe.csv', encoding="utf-8") 

print(df)
print((df.isna().sum() / len(df)) * 100)

###############################################################

# #heatmap
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title("Heatmap des données")
# plt.show()


df2 = df.dropna()

# Sélectionner toutes les colonnes numériques
features = df2.select_dtypes(include=[np.number])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)

# Appliquer PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Afficher le pourcentage de variance expliqué par chaque composante principale
explained_variance = pca.explained_variance_ratio_
print(f"Pourcentage de variance expliqué par chaque composante principale: {explained_variance * 100}")
print(f"Variance totale expliquée par les 2 premières composantes principales: {np.sum(explained_variance) * 100:.2f}%")

# Déterminer le nombre optimal de clusters (Méthode du coude)
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)  # Appliquer KMeans sur les données PCA réduites
    inertia.append(kmeans.inertia_)

# Tracer la méthode du coude
plt.plot(K, inertia, 'bo-')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.title('Méthode du coude')
plt.show()

# Appliquer K-Means avec le k optimal (par exemple, 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df2['Cluster'] = kmeans.fit_predict(data_pca)  # Appliquer K-Means sur les données PCA

# Visualisation des clusters après PCA
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=df2['Cluster'], cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters après PCA et K-Means')
plt.show()