import pandas as pd
import numpy as np

df = pd.read_csv("Datalake_from_OCDE.csv")

################################################################
# Suppression des colonnes contenants que des données vides
# Suppression des colonnes contenants que des données uniques
# Suppression des colonnes contenants des diminutifs
# Mettre la variable TIME_PERIOD en numérique pour faciliter le filtre
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

df = df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE")
taux_vides = df.isnull().mean() * 100

df = df.drop(["Nombre d’abonnés à la télévision par câble", "Total Internet Protocol (IP) telephone subscriptions"], axis=1)
df = df.drop(["Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire)", "Total des recettes des télécommunications"], axis=1)
df = df.drop(["Total des voies d'accès de communication", "Total des abonnements au téléphone cellulaire mobile"], axis=1)

################################################################
# Calcul de la moyenne par pays, uniquement sur les valeurs non nulles
# Remplacer les valeurs nulles par NaN si toutes les valeurs pour une colonne d'un pays sont manquantes

df = df.groupby("Pays").mean(numeric_only=True)
df = df.replace(0, np.nan)
df.to_csv('01_Data_to_use/01_02_Data.csv', encoding="utf-8")


########################################################################################################################################################################
########################################################################################################################################################################
df = pd.read_csv("Datalake_from_OCDE.csv")

df = df.dropna(axis=1)
df = df.loc[:, df.nunique() > 1]
df = df.drop(["COU", "SER"], axis=1)
df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce').astype('Int64')

df = df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE")
df.to_csv("01_Data_to_use/07_08_Dataframe_integral.csv", encoding="utf-8")


########################################################################################################################################################################
########################################################################################################################################################################
df = pd.read_csv("Datalake_from_OCDE.csv")

df = df.dropna(axis=1)
df = df.loc[:, df.nunique() > 1]
df = df.drop(["COU", "SER"], axis=1)
df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce').astype('Int64')
df = df[df['TIME_PERIOD'] > 2004]

df = df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE")
taux_vides = df.isnull().mean() * 100

df = df.drop(["Nombre d’abonnés à la télévision par câble", "Total Internet Protocol (IP) telephone subscriptions"], axis=1)
df = df.drop(["Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire)", "Total des recettes des télécommunications"], axis=1)
df = df.drop(["Total des voies d'accès de communication pour 100 habitants", "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], axis=1)

df = df.dropna()

df.to_csv('01_Data_to_use/03_04_05_06_Dataframe_reglin.csv', encoding="utf-8") 

########################################################################################################################################################################
########################################################################################################################################################################
df = pd.read_csv("Datalake_from_OCDE.csv")

df = df.dropna(axis=1)
df = df.loc[:, df.nunique() > 1]
df = df.drop(["COU", "SER"], axis=1)
df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce').astype('Int64')
df = df[df['TIME_PERIOD'] > 2004]

df = df.pivot(index=["Pays", "TIME_PERIOD"], columns="Série", values="OBS_VALUE")
taux_vides = df.isnull().mean() * 100

df = df.drop(["Nombre d’abonnés à la télévision par câble", "Total Internet Protocol (IP) telephone subscriptions"], axis=1)
df = df.drop(["Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire)", "Total des recettes des télécommunications"], axis=1)
df = df.drop(["Total des voies d'accès de communication pour 100 habitants", "Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], axis=1)

df = df.groupby("Pays").mean(numeric_only=True)
df = df.replace(0, np.nan)

df.to_csv('01_Data_to_use/04_Dataframe_reglin_1.csv', encoding="utf-8") 