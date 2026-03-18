import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA

def approximation(data):
    pt08 = data['PT08.S3(NOx)'].dropna().copy()
    
    print(f"Длина ряда: {len(pt08)}")
    
    df = pd.DataFrame({
        'original': pt08,
        'index': range(len(pt08))
    })
    
    windows = [3, 5, 10, 50]
    
    for window in windows:
        df[f'ma_{window}'] = df['original'].rolling(window=window, center=True).mean()
    
    # График 1: Аппроксимация скользящим средним
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['red', 'green', 'orange', 'purple']
    
    for i, (window, color) in enumerate(zip(windows, colors)):
        ax = axes[i]
        
        ax.plot(df['index'], df['original'], 
                label='Исходный ряд', alpha=0.3, linewidth=0.5, color='blue')
        
        ax.plot(df['index'], df[f'ma_{window}'], 
                label=f'MA окно={window}', linewidth=2, color=color)
        
        ax.set_title(f'Скользящее среднее с окном {window}')
        ax.set_xlabel('Временные шаги')
        ax.set_ylabel('PT08.S3(NOx)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.text(0.02, 0.98, f'Сглаживание: {window} точек', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Аппроксимация скользящим средним с разными окнами', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    train_size = int(len(pt08) * 0.8)
    train, test = pt08[:train_size].values, pt08[train_size:].values
    
    print(f"\nРазмер обучающей выборки: {len(train)}")
    print(f"Размер тестовой выборки: {len(test)}")
    
    max_lags = min(40, len(train) // 4)
    acf_values = acf(train, nlags=max_lags, fft=True)
    
    conf_int = 1.96 / np.sqrt(len(train))
    
    ma_order = 1
    for i in range(1, len(acf_values)):
        if abs(acf_values[i]) > conf_int:
            ma_order = i
        else:
            break
    
    ma_order = min(ma_order, 5)
    
    # График 2: MA модель порядка 5
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ma_model = ARIMA(train, order=(0, 0, 5))
    ma_fitted = ma_model.fit()
    
    predictions = ma_fitted.forecast(steps=len(test))
    
    mse = np.mean((predictions - test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - test))
    
    print(ma_fitted.summary())
    print(f"\nМЕТРИКИ КАЧЕСТВА MA(5):")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    ax.plot(range(len(train)), train, label='Обучающие данные', alpha=0.5, linewidth=1)
    ax.plot(range(len(train), len(train) + len(test)), test, 
            label='Фактические значения (тест)', alpha=0.7, linewidth=1.5)
    ax.plot(range(len(train), len(train) + len(test)), predictions, 
            label='Предсказания MA', color='red', linewidth=2)
    
    ax.set_title('MA модель', fontsize=14)
    ax.set_xlabel('Временные шаги', fontsize=12)
    ax.set_ylabel('PT08.S3(NOx)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ma_fitted

if __name__ == "__main__":
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    approximation(data)