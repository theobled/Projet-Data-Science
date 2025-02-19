import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

np.random.seed(7)

df = pd.read_csv("01_Data_to_use/07_08_Dataframe_integral.csv", encoding='utf-8')

# Filtrage des données
df = df[df["Pays"] == "OCDE - Total"]
target_column = "Total des voies d'accès de communication"
df = df[['TIME_PERIOD', target_column]].dropna()
df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)
df.set_index('TIME_PERIOD', inplace=True)
df = df.sort_index()


# Fonction pour tester la stationnarité et transformer les données
def test_stationarity(series):
    d = 0
    # Test initial de stationnarité
    print("\n Test de stationnarité initial")
    result = adfuller(series)
    print(f"P-value : {result[1]}")
    if result[1] <= 0.05:
        print("Les données sont déjà stationnaires.")
        return series, d

    transformed_series = np.log(series)

    # Test après transformation logarithmique
    result = adfuller(transformed_series)
    print(f"P-value après transformation logarithmique : {result[1]}")
    if result[1] <= 0.05:
        print("Les données sont stationnaires après transformation logarithmique.")
        return transformed_series, d

    # Différenciations successives
    print("\nDifférenciations successives en cours...")
    differentiated_series = transformed_series.diff().dropna()
    d += 1
    while d <= 5:  
        result = adfuller(differentiated_series)
        print(f"Différenciation d'ordre {d} : P-value = {result[1]}")
        if result[1] <= 0.05:
            print(f"Les données sont stationnaires après {d} différenciation(s).")
            return differentiated_series, d
        differentiated_series = differentiated_series.diff().dropna()
        d += 1

    print("Les données ne sont pas stationnaires après 5 différenciations.")
    return differentiated_series, d

# Série cible
time_series = df[target_column]

# Test de stationnarité et transformation
stationary_series, d_order = test_stationarity(time_series)

# Visualisation des ACF et PACF
plt.figure(figsize=(12, 6))

# ACF
plt.subplot(1, 2, 1)
plot_acf(stationary_series, lags=min(20, len(stationary_series) // 2), ax=plt.gca())
plt.title("ACF - Série stationnaire")

# PACF
plt.subplot(1, 2, 2)
plot_pacf(stationary_series, lags=min(20, len(stationary_series) // 2), ax=plt.gca(), method='ywm')
plt.title("PACF - Série stationnaire")

plt.tight_layout()
plt.show()

# Ajustement du modèle ARIMA

p, q = 1, 1
model = ARIMA(time_series, order=(p, d_order, q))
model_fit = model.fit()

print(model_fit.summary())

# Prévisions
forecast_steps = 5
forecast_values = model_fit.forecast(steps=forecast_steps)
forecast_years = range(time_series.index[-1] + 1, time_series.index[-1] + 1 + forecast_steps)
forecast_df = pd.DataFrame({
    "Prévisions": forecast_values.values
}, index=forecast_years)

# Visualisation des prévisions
plt.figure(figsize=(12, 6))
plt.plot(time_series, label="Données réelles")
plt.plot(forecast_df.index, forecast_df["Prévisions"], label="Prévisions", linestyle='--', color='purple')
plt.title("Prévisions ARIMA")
plt.xlabel("Année")
plt.ylabel("Total des voies d'accès de communication")
plt.legend()
plt.grid(alpha=0.5)
plt.show()

# Visualisation des prévisions
residuals = model_fit.resid

plt.figure(figsize=(12, 6))
plt.plot(residuals, label="Résidus", color='blue')
plt.axhline(0, linestyle='--', color='red', alpha=0.7)
plt.title("Résidus du modèle ARIMA")
plt.xlabel("Temps")
plt.ylabel("Résidus")
plt.legend()
plt.grid(alpha=0.5)
plt.show()