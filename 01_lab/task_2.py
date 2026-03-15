import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf


def stationary_row(data):
    pt08 = data['PT08.S3(NOx)'].dropna()

    result = adfuller(pt08)
    print('ADF Statistic (PT08.S3(NOx)): %f' % result[0])
    print('p-value (PT08.S3(NOx)): %f' % result[1])
    print('Critical Values (PT08.S3(NOx)):')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def ar_model(data):
    data = data.sort_values('Datetime').reset_index(drop=True)
    series = data['PT08.S3(NOx)'].dropna().values

    pacf_values = pacf(series, nlags=40, method='ols')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    plot_pacf(series, ax=axes[0], lags=40, method='ols')
    axes[0].set_title('Частичная автокорреляционная функция (PACF)')
    axes[0].set_xlabel('Лаг')
    axes[0].set_ylabel('Частичная автокорреляция')
    axes[0].grid(True, alpha=0.3)
    
    conf_int = 1.96 / np.sqrt(len(series))
    
    significant_lags = []
    for i in range(1, len(pacf_values)):
        if abs(pacf_values[i]) > conf_int:
            significant_lags.append(i)
    
    if significant_lags:
        optimal_lag = min(significant_lags[0], 5)
        print(f"Значимые лаги: {significant_lags[:10]}")
    else:
        optimal_lag = 3
        print("Значимых лагов не найдено, используем lag=3")
    
    print(f"Выбранный лаг для модели: {optimal_lag}")
    
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
       
    model = AutoReg(train, lags=optimal_lag, old_names=False)
    model_fitted = model.fit()

    print(model_fitted.summary())
    
    predictions = model_fitted.predict(start=len(train), end=len(series)-1, dynamic=False)
    
    mse = np.mean((predictions - test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - test))
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    forecast_steps = 20
    forecast = model_fitted.forecast(steps=forecast_steps)
    
    axes[1].plot(range(len(train)), train, label='Обучающие данные', color='blue', alpha=0.7)
    axes[1].plot(range(len(train), len(series)), test, label='Фактические значения (тест)', color='green', alpha=0.7)
    axes[1].plot(range(len(train), len(series)), predictions, label='Предсказания модели', color='red', linestyle='--', alpha=0.7)

    forecast_x = range(len(series), len(series) + forecast_steps)
    axes[1].plot(forecast_x, forecast, label='Прогноз', color='purple', linestyle=':', marker='o', markersize=3)
    
    axes[1].axvline(x=len(train), color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].set_title('AR модель: фактические значения, предсказания и прогноз')
    axes[1].set_xlabel('Временные шаги')
    axes[1].set_ylabel('PT08.S3(NOx)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    stationary_row(data)
    ar_model(data)