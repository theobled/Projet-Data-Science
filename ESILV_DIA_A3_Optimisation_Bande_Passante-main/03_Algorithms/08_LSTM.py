import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import random

seed_test = 41
np.random.seed(seed_test)
tf.random.set_seed(seed_test)
random.seed(seed_test)

df = pd.read_csv("01_Data_to_use/07_08_Dataframe_integral.csv")
df = df[df["Pays"] == "OCDE - Total"]
df = df[["TIME_PERIOD", "Total des voies d'accès de communication"]]
df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)
df.set_index("TIME_PERIOD", inplace=True)

target_variable = "Total des voies d'accès de communication"
data = df[target_variable]

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

#################################### fenêtres temporelles ####################################################
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 20  
X, y = create_dataset(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Construire le modèle LSTM et l'entrainer
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)))  
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

y_pred = model.predict(X_test)

y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

future_steps = 10  
last_window = data_scaled[-time_step:]  
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_window.reshape(1, time_step, 1))
    future_predictions.append(prediction[0, 0])
    last_window = np.append(last_window[1:], prediction, axis=0)  # Glissement de la fenêtre

future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Générer les années 
future_years = np.arange(df.index[-1] + 1, df.index[-1] + 1 + future_steps)

# Vérifier la continuité entre les données réelles et les prédictions futures
last_real = data.values[-1]
first_future_pred = future_predictions_rescaled[0][0]
print(f"\nDernière valeur réelle : {last_real:.2f}")
print(f"Première prédiction future : {first_future_pred:.2f}")


# Ajuster les prédictions futures pour commencer à la dernière valeur réelle
adjusted_future_predictions = [last_real]
for pred in future_predictions_rescaled.flatten():
    adjusted_future_predictions.append(adjusted_future_predictions[-1] + (pred - adjusted_future_predictions[-1]))

adjusted_future_predictions = np.array(adjusted_future_predictions)

plt.figure(figsize=(12, 8))
plt.plot(df.index, data.values, label="Données réelles", color="blue")
adjusted_combined_years = np.append(df.index[-1:], future_years)

plt.plot(adjusted_combined_years, adjusted_future_predictions, label="Prédictions futures ajustées", color="red")
plt.title("Prédictions des voies d'accès de communication (1996-2018 et au-delà)")
plt.xlabel("Année")
plt.ylabel("Total des voies d'accès de communication")
plt.legend()
plt.show()

# coef d'evalution
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
