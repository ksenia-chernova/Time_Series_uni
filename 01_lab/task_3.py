import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

def approximation(data):
    # Получаем ряд и удаляем пропуски
    pt08 = data['PT08.S3(NOx)'].dropna().copy()
    
    print(f"Длина ряда: {len(pt08)}")
    print(f"Первые 5 значений:\n{pt08.head()}")
    
    # Создаем DataFrame для удобства работы
    df = pd.DataFrame({
        'original': pt08,
        'index': range(len(pt08))
    })
    
    # Рассчитываем скользящие средние с разными окнами
    windows = [3, 5, 10, 50]
    
    for window in windows:
        df[f'ma_{window}'] = df['original'].rolling(window=window, center=True).mean()
    
    # 1. ГРАФИК АППРОКСИМАЦИИ СКОЛЬЗЯЩИМИ СРЕДНИМИ
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['red', 'green', 'orange', 'purple']
    
    for i, (window, color) in enumerate(zip(windows, colors)):
        ax = axes[i]
        
        # Рисуем исходный ряд (полупрозрачный)
        ax.plot(df['index'], df['original'], 
                label='Исходный ряд', alpha=0.3, linewidth=0.5, color='blue')
        
        # Рисуем скользящее среднее
        ax.plot(df['index'], df[f'ma_{window}'], 
                label=f'MA окно={window}', linewidth=2, color=color)
        
        ax.set_title(f'Скользящее среднее с окном {window}')
        ax.set_xlabel('Временные шаги')
        ax.set_ylabel('PT08.S3(NOx)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем текст с информацией
        ax.text(0.02, 0.98, f'Сглаживание: {window} точек', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Аппроксимация скользящим средним с разными окнами', fontsize=14)
    plt.tight_layout()
    plt.show()
       
    # 3. МОДЕЛЬ СКОЛЬЗЯЩЕГО СРЕДНЕГО (MA) ДЛЯ ПРЕДСКАЗАНИЯ
    
    # Разделяем данные на обучающую и тестовую выборки
    train_size = int(len(pt08) * 0.8)
    train, test = pt08[:train_size].values, pt08[train_size:].values
    
    print(f"\nРазмер обучающей выборки: {len(train)}")
    print(f"Размер тестовой выборки: {len(test)}")
    
    # Анализ ACF для определения порядка MA модели
    max_lags = min(40, len(train) // 4)
    acf_values = acf(train, nlags=max_lags, fft=True)
    
    # Визуализация ACF
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График ACF
    plot_acf(train, ax=axes[0, 0], lags=max_lags)
    axes[0, 0].set_title('ACF для определения порядка MA модели')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Определяем порядок MA (где ACF резко обрывается)
    conf_int = 1.96 / np.sqrt(len(train))
    
    ma_order = 1
    for i in range(1, len(acf_values)):
        if abs(acf_values[i]) > conf_int:
            ma_order = i
        else:
            # Первый незначимый лаг - порядок MA
            break
    
    ma_order = min(ma_order, 5)  # Ограничиваем до 5
    print(f"Определенный порядок MA модели: {ma_order}")
    
    # Для MA модели используем AutoReg с максимальным лагом и смотрим на коэффициенты
    # Но фактически нам нужна MA модель, которую можно построить через ARMA
    from statsmodels.tsa.arima.model import ARIMA
    
    # Обучаем MA модель (это ARIMA с order=(0,0,q))
    ma_model = ARIMA(train, order=(0, 0, ma_order))
    ma_fitted = ma_model.fit()
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ MA МОДЕЛИ:")
    print("="*50)
    print(ma_fitted.summary())
    
    # Предсказания на тестовой выборке
    predictions = ma_fitted.forecast(steps=len(test))
    
    # Оценка качества
    mse = np.mean((predictions - test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - test))
    
    print(f"\nМЕТРИКИ КАЧЕСТВА MA({ma_order}):")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # График предсказаний MA модели
    ax = axes[0, 1]
    ax.plot(range(len(train)), train, label='Обучающие данные', alpha=0.7)
    ax.plot(range(len(train), len(train) + len(test)), test, 
            label='Фактические значения (тест)', alpha=0.7)
    ax.plot(range(len(train), len(train) + len(test)), predictions, 
            label=f'Предсказания MA({ma_order})', linestyle='--', marker='o', markersize=3)
    ax.axvline(x=len(train), color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title(f'MA модель порядка {ma_order}')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Сравнение разных порядков MA
    ax = axes[1, 0]
    ma_orders_to_try = [1, 2, 3, 5]
    
    for q in ma_orders_to_try:
        model = ARIMA(train, order=(0, 0, q))
        fitted = model.fit()
        pred = fitted.forecast(steps=len(test))
        
        mse_q = np.mean((pred - test) ** 2)
        rmse_q = np.sqrt(mse_q)
        
        ax.plot(range(len(train), len(train) + len(test)), pred, 
                label=f'MA({q}) - RMSE={rmse_q:.2f}', linestyle='--', marker='.', markersize=2)
    
    ax.plot(range(len(train), len(train) + len(test)), test, 
            label='Фактические значения', linewidth=2, color='black', alpha=0.7)
    ax.axvline(x=len(train), color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title('Сравнение MA моделей разных порядков')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Прогноз на будущие 24 точки
    ax = axes[1, 1]
    forecast_steps = 24
    
    # Используем лучшую модель (с минимальным RMSE)
    best_rmse = float('inf')
    best_q = ma_order
    best_forecast = None
    
    for q in ma_orders_to_try:
        model = ARIMA(train, order=(0, 0, q))
        fitted = model.fit()
        pred = fitted.forecast(steps=len(test))
        rmse_q = np.sqrt(np.mean((pred - test) ** 2))
        
        if rmse_q < best_rmse:
            best_rmse = rmse_q
            best_q = q
            # Переобучаем на всех данных для прогноза
            final_model = ARIMA(pt08.values, order=(0, 0, q))
            final_fitted = final_model.fit()
            best_forecast = final_fitted.forecast(steps=forecast_steps)
    
    # Визуализация прогноза
    ax.plot(range(len(pt08)), pt08.values, label='Исторические данные', alpha=0.5)
    forecast_x = range(len(pt08), len(pt08) + forecast_steps)
    ax.plot(forecast_x, best_forecast, label=f'Прогноз MA({best_q})', 
            color='red', linewidth=2, marker='o', markersize=4)
    ax.axvline(x=len(pt08), color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title(f'Прогноз на {forecast_steps} шагов вперед (MA({best_q}))')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. ДОПОЛНИТЕЛЬНО: Сравнение скользящего среднего и MA модели
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Исходный ряд
    ax.plot(df['index'], df['original'], 
            label='Исходный ряд', alpha=0.2, linewidth=0.5, color='gray')
    
    # Скользящее среднее с окном 10
    ax.plot(df['index'], df['ma_10'], 
            label='Скользящее среднее (окно=10)', linewidth=1.5, color='blue')
    
    # Предсказания MA модели (на тестовом участке)
    test_indices = range(len(train), len(train) + len(test))
    ax.plot(test_indices, predictions, 
            label=f'MA модель порядка {ma_order}', linewidth=2, color='red', linestyle='--')
    
    ax.set_title('Сравнение: Скользящее среднее (сглаживание) vs MA модель (предсказание)')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print("\n" + "="*50)
    print("ИТОГИ:")
    print("="*50)
    print(f"1. Скользящие средние (окна {windows}) - для сглаживания ряда")
    print(f"2. Оптимальный порядок MA модели: {best_q}")
    print(f"3. Точность MA модели: RMSE={best_rmse:.2f}")
    print(f"4. Прогноз на {forecast_steps} шагов: первые 5 значений {best_forecast[:5]}")

if __name__ == "__main__":
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    approximation(data)