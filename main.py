# Import necessary libraries
from prophet import Prophet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.metrics import mean_squared_error
df = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Facebook-prophet\AirPassengers.csv")
print(df.head())
df = df.rename(columns={'#Passengers' : 'y','Month' : 'ds'})
df['y_orig'] = df['y']
df['y'] = np.log(df['y'])
print(df.head())
# Initiate model
model = Prophet()
model.fit(df)
# Future data
future_data = model.make_future_dataframe(periods=12,freq='ME')
forecast_data = model.predict(future_data)
print(forecast_data.tail())
model.plot(forecast_data)
plt.show()
model.plot_components(forecast_data)
plt.show()
forecast_data_orig = forecast_data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
model.plot(forecast_data_orig)
plt.show()
df['y_log'] = df['y']
df['y'] = df['y_orig']
final_df = pd.DataFrame(forecast_data_orig)
# Plot predicted an actual line graph with X=dates and y=Outbased
actual_chart = go.Scatter(y=df['y_orig'],name='Actual')
predict_chart = go.Scatter(y=final_df['yhat'],name='Predicted')
predict_chart_upper = go.Scatter(y=final_df['yhat_upper'],name='Predicted Upper')
predict_chart_lower = go.Scatter(y=final_df['yhat_lower'],name='Predicted Lower')
py.plot([actual_chart,predict_chart,predict_chart_upper,predict_chart_lower],image_width=400,image_height=400)

