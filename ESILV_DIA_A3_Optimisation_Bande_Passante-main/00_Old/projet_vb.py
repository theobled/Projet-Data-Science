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
# Filtrer les lignes où TIME_PERIOD est supérieur à 2004

df = df.dropna(axis=1)
df = df.loc[:, df.nunique() > 1]
df = df.drop(["COU", "SER"], axis=1)
df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce').astype('Int64')
df = df[df['TIME_PERIOD'] > 2004]

################################################################
# Pivoter les colonnes "Série" pour les avoir en colonne
# Calcule du taux des valeurs vides dans chaque colonne
# Suppression des colonnes où la valeur de vide est supérieur à 50% 
# Suppression des colonnes où la monnaie est en devise locale pour garder leurs équivalents en dollars
# Pas de correlation sur heatmap et c'est limite équivalent au total

df = df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE")
taux_vides = df.isnull().mean() * 100
print(taux_vides)

df = df.drop(["Nombre d’abonnés à la télévision par câble", "Total Internet Protocol (IP) telephone subscriptions"], axis=1)
df = df.drop(["Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire)", "Total des recettes des télécommunications"], axis=1)
df = df.drop(["Total des voies d'accès de communication pour 100 habitants", "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], axis=1)

df2 = df.dropna()
df2.to_csv('Dataframe_v4(Mehdi).csv', encoding="utf-8") 
################################################################
# Calcul de la moyenne par pays, on somme chaque colonne et divise par 14 (nombre d'années par pays entre 2005 et 2018)
# Remplacer les valeurs nulles par NaN si toutes les valeurs pour une colonne d'un pays sont manquantes
df = df.groupby("Pays").sum(numeric_only=True) / 14
df = df.replace(0, np.nan)

################################################################
df.to_csv('Dataframe_v3.csv', encoding="utf-8") 

taux_vides = df.isnull().mean() * 100
print(taux_vides)