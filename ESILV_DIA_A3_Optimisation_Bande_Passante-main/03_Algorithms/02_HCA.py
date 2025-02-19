import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df = pd.read_csv('01_Data_to_use/01_02_Data.csv')
df = df[~df['Pays'].isin(['OCDE - Total'])]

x_col = 'Total des abonnements au téléphone cellulaire mobile pour 100 habitants'
y_col = 'Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire) USD'

filtered_df = df[[x_col, y_col, 'Pays']].dropna()
data = filtered_df[[x_col, y_col]] 

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

linked = linkage(data_scaled, method='complete') 

################################### Dendrogramme #######################################

plt.figure(figsize=(12, 8))
dendrogram(linked)
plt.title('Dendrogramme pour HCA')
plt.xlabel('Index des pays')
plt.ylabel('Distance')
plt.show()

########################################################################################

clusters = fcluster(linked, t=3, criterion='maxclust') 
filtered_df['Cluster'] = clusters

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
plt.title('Clusters Hiérarchiques sur les télécommunications')
plt.legend()
plt.grid(True)
plt.show()
