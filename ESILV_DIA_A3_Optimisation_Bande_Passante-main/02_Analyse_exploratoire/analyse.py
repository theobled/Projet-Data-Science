import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("01_Data_to_use/07_08_Dataframe_integral.csv")
df.set_index(['Pays', 'TIME_PERIOD'], inplace=True)
df = df.drop(["Nombre d’abonnés à la télévision par câble", "Total Internet Protocol (IP) telephone subscriptions"], axis=1)
df = df.drop(["Investissements totaux dans les télécommunications (pour lignes fixes et réseau mobile cellulaire)", "Total des recettes des télécommunications"], axis=1)


df.reset_index(inplace=True)

# Boxplot pour la variable "Total des voies d'accès de communication" (inclut OCDE - Total et États-Unis)
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Pays", y="Total des voies d'accès de communication")
plt.xticks(rotation=90)
plt.title("Boxplot - Total des voies d'accès de communication par pays")
plt.ylabel("Total des voies d'accès de communication")
plt.xlabel("Pays")
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()


df_filtered = df[~df["Pays"].isin(["OCDE - Total", "États-Unis"])]

# Boxplot pour la variable "Total des voies d'accès de communication"
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_filtered, x="Pays", y="Total des voies d'accès de communication")
plt.xticks(rotation=90)
plt.title("Boxplot - Total des voies d'accès de communication par pays (sans OCDE - Total et États-Unis)")
plt.ylabel("Total des voies d'accès de communication")
plt.xlabel("Pays")
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

# Boxplot pour la variable "Total des recettes des télécommunications USD"
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_filtered, x="Pays", y="Total des recettes des télécommunications USD")
plt.xticks(rotation=90)
plt.title("Boxplot - Total des recettes des télécommunications USD par pays (sans OCDE - Total et États-Unis)")
plt.ylabel("Total des recettes des télécommunications USD")
plt.xlabel("Pays")
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# Tableau résumé graphique
# ---------------------------
df_summary = df_filtered.dropna() 
summary = df_summary.describe()


plt.figure(figsize=(14, 8))  
sns.heatmap(summary, annot=True, fmt=".2f", cmap="Blues", cbar=False, linewidths=0.5)
plt.title("Résumé statistique des variables", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10) 
plt.yticks(fontsize=10) 
plt.tight_layout()
plt.show()

# ---------------------------
# Heatmap
# ---------------------------

df_corr = df.drop(columns=["Pays", "TIME_PERIOD"])
correlation_matrix = df_corr.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, square=True)
plt.title("Heatmap de corrélation des variables numériques", fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotation des labels en bas
plt.yticks(rotation=0)  # Labels à gauche non inclinés
plt.tight_layout()
plt.show()

# ---------------------------
# Tendance lignes fixes
# ---------------------------

df_ocde = df[df["Pays"] == "OCDE - Total"]
df_grouped_ocde = df_ocde.groupby("TIME_PERIOD")["Total des lignes d'accès téléphoniques"].sum()

plt.figure(figsize=(12, 6))
plt.plot(df_grouped_ocde.index, df_grouped_ocde.values, marker='o', linestyle='-', color='green')
plt.title("Évolution du Total des lignes d'accès téléphoniques pour OCDE - Total", fontsize=16)
plt.xlabel("Année", fontsize=12)
plt.ylabel("Total des lignes d'accès téléphoniques", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# Tendance cartes prépayées
# ---------------------------
df_grouped_ocde_prepaid = df_ocde.groupby("TIME_PERIOD")["Abonnements au téléphone cellulaire mobile utilisant des cartes prépayés"].sum()

plt.figure(figsize=(12, 6))
plt.plot(df_grouped_ocde_prepaid.index, df_grouped_ocde_prepaid.values, marker='o', linestyle='-', color='blue')
plt.title("Évolution des abonnements aux cartes prépayées pour OCDE - Total", fontsize=16)
plt.xlabel("Année", fontsize=12)
plt.ylabel("Total des abonnements aux cartes prépayées", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Pays", y="Total des voies d'accès de communication pour 100 habitants")
plt.xticks(rotation=90)
plt.title("Boxplot - Total des voies d'accès de communication pour 100 habitants par pays", fontsize=16)
plt.xlabel("Pays", fontsize=12)
plt.ylabel("Total des voies d'accès de communication pour 100 habitants", fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()