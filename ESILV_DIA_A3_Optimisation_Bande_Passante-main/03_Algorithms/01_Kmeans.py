import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('01_Data_to_use/01_02_Data.csv')
df = df[~df['Pays'].isin(['OCDE - Total'])]

x_col = 'Total des abonnements au téléphone cellulaire mobile pour 100 habitants'
y_col = 'Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD'

filtered_df = df[[x_col, y_col, 'Pays']].dropna()  
data = filtered_df[[x_col, y_col]]  

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

################################### Méthode du coude Elbow #########################################

wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_) 

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

####################################################################################################

kmeans = KMeans(n_clusters=4, random_state=42)
filtered_df['Cluster'] = kmeans.fit_predict(data_scaled)

plt.figure(figsize=(12, 8))
for cluster in filtered_df['Cluster'].unique():
    cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
    plt.scatter(
        cluster_data[x_col],
        cluster_data[y_col],
        label=f'Cluster {cluster}',
        s=50  
    )
    for _, row in cluster_data.iterrows():
        plt.text(
            row[x_col],
            row[y_col],
            row['Pays'], 
            fontsize=8,
            alpha=0.7
        )

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title('Clusters sur les télécommunications')
plt.legend()
plt.grid(True)
plt.show()