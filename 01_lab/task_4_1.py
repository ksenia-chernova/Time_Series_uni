import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def arma_1(data):
    df = data[['Datetime', 'PT08.S3(NOx)']].copy()
    df = df.dropna().reset_index(drop=True)
    df['PT08.S3(NOx)'] = df['PT08.S3(NOx)'].astype(str).str.replace(',', '.').astype(float)
    df.set_index('Datetime', inplace=True)

    # p — порядок авторегрессии для стационарного ряда;
    # d — порядок дифференцирования (сколько раз брали разность);
    # q — порядок скользящего среднего.  
    p, q = 2, 2
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    model = ARIMA(train, order=(p, 0, q))
    model_fit = model.fit()
    
    forecast_result = model_fit.get_forecast(steps=len(test))
    predicted_values = forecast_result.predicted_mean
    
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['PT08.S3(NOx)'], label='Обучающие данные', alpha=0.5, linewidth=1)
    plt.plot(test.index, test['PT08.S3(NOx)'], label='Фактические значения', alpha=0.7, linewidth=1.5)
    plt.plot(test.index, predicted_values, label='Прогноз ARMA(2,2)', color='red', linewidth=2)
    
    plt.title('Прогнозирование временного ряда PT08.S3(NOx) с помощью ARMA(2,2)', fontsize=14)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Значение PT08.S3(NOx)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"Модель ARMA({p},{q})")
    print(model_fit.summary())
    
    mse = np.mean((test['PT08.S3(NOx)'].values - predicted_values) ** 2)
    mae = np.mean(np.abs(test['PT08.S3(NOx)'].values - predicted_values))
    rmse = np.sqrt(mse)
    
    print(f"\nМетрики качества прогноза:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # MSE (среднеквадратическая ошибка, Mean Squared Error)
    # MAE (средняя абсолютная ошибка, Mean Absolute Error)
    # RMSE (корень из среднеквадратичной ошибки, Root Mean Squared Error) 
    
    return model_fit

if __name__ == "__main__":
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    model = arma_1(data)