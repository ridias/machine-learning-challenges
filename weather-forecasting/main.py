import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

data = pd.read_csv("DailyDelhiClimateTrain.csv")
print(data.head())

print(data.describe())


# preprocess bad values.
for i in range(0, len(data['meantemp'])):
    if data['meantemp'][i] >= 1000:
        data.at[i, 'meantemp'] = data['meantemp'][i] / 1000

for i in range(0, len(data['humidity'])):
    if data['humidity'][i] >= 1000:
        data.at[i, 'humidity'] = data['humidity'][i] / 1000

        
for i in range(0, len(data['wind_speed'])):
    if data['wind_speed'][i] >= 300:
        data.at[i, 'wind_speed'] = data['wind_speed'][i] / 1000

for i in range(0, len(data['meanpressure'])):
    if data['meanpressure'][i] >= 2000:
        data.at[i, 'meanpressure'] = data['meanpressure'][i] / 1000
 
print(data.describe())

print(data.info())

plt.plot(data['date'], data['meantemp'])
plt.xticks(rotation=45)  # Rotar las fechas para una mejor legibilidad
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
plt.show()

plt.plot(data['date'], data['humidity'])
plt.xticks(rotation=45)  # Rotar las fechas para una mejor legibilidad
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
plt.show()

plt.plot(data['date'], data['wind_speed'])
plt.xticks(rotation=45)  # Rotar las fechas para una mejor legibilidad
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
plt.show()

plt.plot(data['date'], data['meanpressure'])
plt.xticks(rotation=45)  # Rotar las fechas para una mejor legibilidad
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
plt.show()

forecast_data = data.rename(columns = {"date": "ds", 
                                       "meantemp": "y"})
print(forecast_data)

model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions).show()