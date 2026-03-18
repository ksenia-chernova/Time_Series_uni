import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def arma_manual_prediction(data, n_predict=100):
    df = data[['Datetime', 'PT08.S3(NOx)']].copy()
    df = df.dropna().reset_index(drop=True)
    df['PT08.S3(NOx)'] = df['PT08.S3(NOx)'].astype(str).str.replace(',', '.').astype(float)
    df.set_index('Datetime', inplace=True)
    
    model = ARIMA(df['PT08.S3(NOx)'], order=(2, 0, 2))
    model_fit = model.fit()
    
    ar_params = np.array([0.6, 0.2])      # φ1, φ2
    ma_params = np.array([0.3, 0.1])      # θ1, θ2
    const = 800  # примерно среднее значение ряда
    
    residuals = model_fit.resid.values
    
    print(f"Коэффициенты AR(2): φ1 = {ar_params[0]:.4f}, φ2 = {ar_params[1]:.4f}")
    print(f"Коэффициенты MA(2): θ1 = {ma_params[0]:.4f}, θ2 = {ma_params[1]:.4f}")
    print(f"Константа: c = {const:.4f}")
    
    series = df['PT08.S3(NOx)'].values
    last_values = series[-2:].copy()
    last_residuals = residuals[-2:].copy()

    np.random.seed(42)
    future_residuals = np.random.normal(0, np.std(residuals), n_predict)
    
    predictions = []
    
    for i in range(n_predict):
        ar_part = const + ar_params[0] * last_values[-1] + ar_params[1] * last_values[-2]
        ma_part = future_residuals[i] + ma_params[0] * last_residuals[-1] + ma_params[1] * last_residuals[-2]
        
        pred = ar_part + ma_part
        predictions.append(pred)
        
        last_values = np.append(last_values[1:], pred)
        last_residuals = np.append(last_residuals[1:], future_residuals[i])
    
    forecast_result = model_fit.get_forecast(steps=n_predict)
    library_pred = forecast_result.predicted_mean.values
    
    last_date = df.index[-1]
    time_diff = df.index[-1] - df.index[-2]
    future_index = pd.date_range(start=last_date, periods=n_predict + 1, freq=time_diff)[1:]
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(df.index[-200:], df['PT08.S3(NOx)'].values[-200:], 
             label='Исходные данные', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(future_index, predictions, label='Ручная реализация ARMA(2,2)', color='red', linewidth=2)
    plt.plot(future_index, library_pred, label='Библиотечная реализация', color='green', linewidth=2, alpha=0.7)
    
    plt.title('Сравнение ручной и библиотечной реализаций ARMA(2,2) с исходными данными', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Значение PT08.S3(NOx)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return predictions, future_index

if __name__ == "__main__":
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    
    predictions, future_dates = arma_manual_prediction(data, n_predict=100)