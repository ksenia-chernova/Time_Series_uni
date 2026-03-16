import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import warnings
warnings.filterwarnings('ignore')

class ARMA_Manual:
    """
    Самостоятельная реализация ARMA модели
    ARMA(p,q): X_t = c + ε_t + Σ(φ_i * X_{t-i}) + Σ(θ_j * ε_{t-j})
    """
    
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.phi = None  # AR коэффициенты
        self.theta = None  # MA коэффициенты
        self.c = None  # константа
        self.residuals = None
        self.n = None
        
    def fit(self, series, max_iter=1000):
        """
        Обучение модели методом максимального правдоподобия
        """
        self.n = len(series)
        self.series = series.copy()
        
        # Начальные приближения
        initial_params = np.zeros(self.p + self.q + 2)  # +2 для c и sigma
        initial_params[-1] = np.std(series)  # sigma
        
        # Оптимизация
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(series,),
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )
        
        # Сохраняем коэффициенты
        params = result.x
        self.c = params[0]
        self.phi = params[1:1+self.p] if self.p > 0 else np.array([])
        self.theta = params[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
        self.sigma = params[-1]
        
        # Вычисляем остатки
        self.residuals = self._compute_residuals(series)
        
        return self
    
    def _negative_log_likelihood(self, params, series):
        """
        Отрицательное логарифмическое правдоподобие
        """
        n = len(series)
        c = params[0]
        phi = params[1:1+self.p] if self.p > 0 else np.array([])
        theta = params[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
        sigma = params[-1]
        
        residuals = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            # AR часть
            ar_part = c
            for i in range(1, self.p + 1):
                if t - i >= 0 and i <= len(phi):
                    ar_part += phi[i-1] * series[t-i]
            
            # MA часть
            ma_part = 0
            for j in range(1, self.q + 1):
                if t - j >= 0 and j <= len(theta) and not np.isnan(residuals[t-j]):
                    ma_part += theta[j-1] * residuals[t-j]
            
            # Предсказание
            prediction = ar_part + ma_part
            
            # Остаток
            residuals[t] = series[t] - prediction
        
        # Логарифмическое правдоподобие (нормальное распределение)
        valid_residuals = residuals[max(self.p, self.q):]
        n_valid = len(valid_residuals)
        
        if n_valid == 0:
            return 1e10
        
        log_likelihood = -0.5 * n_valid * np.log(2 * np.pi * sigma**2) - \
                         np.sum(valid_residuals**2) / (2 * sigma**2)
        
        return -log_likelihood  # минимизируем отрицательное правдоподобие
    
    def _compute_residuals(self, series):
        """
        Вычисление остатков модели
        """
        n = len(series)
        residuals = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            # AR часть
            ar_part = self.c
            for i in range(1, self.p + 1):
                if t - i >= 0 and i <= len(self.phi):
                    ar_part += self.phi[i-1] * series[t-i]
            
            # MA часть
            ma_part = 0
            for j in range(1, self.q + 1):
                if t - j >= 0 and j <= len(self.theta) and not np.isnan(residuals[t-j]):
                    ma_part += self.theta[j-1] * residuals[t-j]
            
            # Предсказание
            prediction = ar_part + ma_part
            
            # Остаток
            residuals[t] = series[t] - prediction
        
        return residuals
    
    def predict(self, steps, last_values=None, last_residuals=None):
        """
        Предсказание на steps шагов вперед
        """
        if last_values is None:
            last_values = self.series[-self.p:] if self.p > 0 else []
        
        if last_residuals is None:
            last_residuals = self.residuals[-self.q:] if self.q > 0 else []
        
        predictions = []
        current_values = list(last_values)
        current_residuals = list(last_residuals)
        
        for step in range(steps):
            # AR часть
            ar_part = self.c
            for i in range(1, self.p + 1):
                if i <= len(current_values):
                    ar_part += self.phi[i-1] * current_values[-i]
            
            # MA часть (используем последние остатки)
            ma_part = 0
            for j in range(1, self.q + 1):
                if j <= len(current_residuals):
                    ma_part += self.theta[j-1] * current_residuals[-j]
            
            # Предсказание
            prediction = ar_part + ma_part
            
            # Добавляем шум для реалистичности (опционально)
            # noise = np.random.normal(0, self.sigma)
            # prediction += noise
            
            predictions.append(prediction)
            
            # Обновляем значения для следующего шага
            current_values.append(prediction)
            if self.q > 0:
                # Остаток для следующего шага (в реальности мы его не знаем)
                current_residuals.append(0)  # Используем 0 как ожидаемое значение остатка
        
        return np.array(predictions)
    
    def summary(self):
        """
        Краткая информация о модели
        """
        print(f"\nARMA({self.p},{self.q}) модель:")
        print(f"Константа (c): {self.c:.4f}")
        if self.p > 0:
            print(f"AR коэффициенты (φ): {self.phi}")
        if self.q > 0:
            print(f"MA коэффициенты (θ): {self.theta}")
        print(f"Сигма (σ): {self.sigma:.4f}")
        print(f"Количество наблюдений: {self.n}")


def determine_arma_order(series, max_p=5, max_q=5):
    """
    Определение порядка ARMA модели по ACF и PACF
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # ACF и PACF
    plot_acf(series, ax=axes[0, 0], lags=40)
    axes[0, 0].set_title('ACF')
    axes[0, 0].grid(True, alpha=0.3)
    
    plot_pacf(series, ax=axes[0, 1], lags=40, method='ols')
    axes[0, 1].set_title('PACF')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Автоматическое определение порядка по информационным критериям
    from statsmodels.tsa.arima.model import ARIMA
    
    results = []
    best_aic = float('inf')
    best_order = (1, 0, 1)
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(series, order=(p, 0, q))
                fitted = model.fit()
                aic = fitted.aic
                results.append((p, q, aic))
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, 0, q)
            except:
                continue
    
    # Таблица результатов
    ax = axes[1, 0]
    ax.axis('tight')
    ax.axis('off')
    
    cell_text = [[f"ARMA({p},{q})", f"{aic:.1f}"] for p, q, aic in results[:10]]
    ax.table(cellText=cell_text, colLabels=['Модель', 'AIC'], 
             loc='center', cellLoc='center')
    ax.set_title('Топ-10 моделей по AIC')
    
    # Визуализация AIC
    ax = axes[1, 1]
    p_values = [r[0] for r in results]
    q_values = [r[1] for r in results]
    aic_values = [r[2] for r in results]
    
    scatter = ax.scatter(p_values, q_values, c=aic_values, cmap='viridis', 
                        s=100, alpha=0.7)
    ax.set_xlabel('p (AR порядок)')
    ax.set_ylabel('q (MA порядок)')
    ax.set_title('AIC для различных (p,q)')
    plt.colorbar(scatter, ax=ax, label='AIC')
    
    # Отметим лучшую модель
    ax.scatter(best_order[0], best_order[2], color='red', s=200, 
              marker='*', label=f'Лучшая: ARMA{best_order}')
    ax.legend()
    
    plt.suptitle(f'Определение порядка ARMA модели\nЛучшая: ARMA{best_order} с AIC={best_aic:.1f}')
    plt.tight_layout()
    plt.show()
    
    print(f"\nРекомендуемый порядок: ARMA{best_order} (AIC={best_aic:.1f})")
    return best_order[0], best_order[2]


def arma_prediction(data):
    """
    Основная функция для ARMA предсказания
    """
    # Получаем ряд и удаляем пропуски
    pt08 = data['PT08.S3(NOx)'].dropna().copy()
    
    print(f"Длина ряда: {len(pt08)}")
    print(f"Статистика ряда:")
    print(f"  Среднее: {pt08.mean():.2f}")
    print(f"  Std: {pt08.std():.2f}")
    print(f"  Мин: {pt08.min():.2f}")
    print(f"  Макс: {pt08.max():.2f}")
    
    # Нормализация для лучшей сходимости
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pt08_scaled = scaler.fit_transform(pt08.values.reshape(-1, 1)).flatten()
    
    # Разделяем данные
    train_size = int(len(pt08_scaled) * 0.8)
    train_scaled = pt08_scaled[:train_size]
    test_scaled = pt08_scaled[train_size:]
    train_original = pt08.values[:train_size]
    test_original = pt08.values[train_size:]
    
    print(f"\nРазмер обучающей выборки: {len(train_scaled)}")
    print(f"Размер тестовой выборки: {len(test_scaled)}")
    
    # Определяем порядок модели
    p, q = determine_arma_order(train_scaled, max_p=4, max_q=4)
    
    # ЧАСТЬ 1: Модель с использованием statsmodels (для сравнения)
    print("\n" + "="*60)
    print("МОДЕЛЬ ARMA С ИСПОЛЬЗОВАНИЕМ STATSMODELS")
    print("="*60)
    
    sm_model = ARIMA(train_scaled, order=(p, 0, q))
    sm_fitted = sm_model.fit()
    print(sm_fitted.summary())
    
    # Предсказания statsmodels
    sm_pred_scaled = sm_fitted.forecast(steps=len(test_scaled))
    sm_pred_original = scaler.inverse_transform(sm_pred_scaled.reshape(-1, 1)).flatten()
    
    # Метрики statsmodels
    sm_rmse = np.sqrt(np.mean((sm_pred_original - test_original) ** 2))
    sm_mae = np.mean(np.abs(sm_pred_original - test_original))
    
    print(f"\nМетрики statsmodels ARMA({p},{q}):")
    print(f"  RMSE: {sm_rmse:.2f}")
    print(f"  MAE: {sm_mae:.2f}")
    
    # ЧАСТЬ 2: Самостоятельная реализация ARMA
    print("\n" + "="*60)
    print("САМОСТОЯТЕЛЬНАЯ РЕАЛИЗАЦИЯ ARMA")
    print("="*60)
    
    manual_model = ARMA_Manual(p, q)
    manual_model.fit(train_scaled)
    manual_model.summary()
    
    # Предсказания manual модели
    manual_pred_scaled = manual_model.predict(steps=len(test_scaled))
    manual_pred_original = scaler.inverse_transform(manual_pred_scaled.reshape(-1, 1)).flatten()
    
    # Метрики manual модели
    manual_rmse = np.sqrt(np.mean((manual_pred_original - test_original) ** 2))
    manual_mae = np.mean(np.abs(manual_pred_original - test_original))
    
    print(f"\nМетрики manual ARMA({p},{q}):")
    print(f"  RMSE: {manual_rmse:.2f}")
    print(f"  MAE: {manual_mae:.2f}")
    
    # Визуализация результатов
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # 1. Исходный ряд
    ax = axes[0, 0]
    ax.plot(range(len(train_original)), train_original, 
            label='Обучающие данные', alpha=0.7, linewidth=0.5)
    ax.plot(range(len(train_original), len(train_original) + len(test_original)), 
            test_original, label='Тестовые данные', alpha=0.7, linewidth=0.5)
    ax.set_title('Исходный временной ряд')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Statsmodels предсказания
    ax = axes[0, 1]
    ax.plot(range(len(train_original)), train_original, 
            label='Обучающие', alpha=0.5, linewidth=0.5)
    ax.plot(range(len(train_original), len(train_original) + len(test_original)), 
            test_original, label='Фактические', alpha=0.7, linewidth=1)
    ax.plot(range(len(train_original), len(train_original) + len(test_original)), 
            sm_pred_original, label=f'Statsmodels ARMA({p},{q})', 
            linewidth=2, linestyle='--', color='red')
    ax.axvline(x=len(train_original), color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title(f'Statsmodels ARMA({p},{q}) предсказания\nRMSE={sm_rmse:.2f}')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Manual предсказания
    ax = axes[1, 0]
    ax.plot(range(len(train_original)), train_original, 
            label='Обучающие', alpha=0.5, linewidth=0.5)
    ax.plot(range(len(train_original), len(train_original) + len(test_original)), 
            test_original, label='Фактические', alpha=0.7, linewidth=1)
    ax.plot(range(len(train_original), len(train_original) + len(test_original)), 
            manual_pred_original, label=f'Manual ARMA({p},{q})', 
            linewidth=2, linestyle='--', color='green')
    ax.axvline(x=len(train_original), color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title(f'Manual ARMA({p},{q}) предсказания\nRMSE={manual_rmse:.2f}')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Сравнение предсказаний
    ax = axes[1, 1]
    ax.plot(range(len(test_original)), test_original, 
            label='Фактические', linewidth=2, color='black')
    ax.plot(range(len(test_original)), sm_pred_original, 
            label=f'Statsmodels (RMSE={sm_rmse:.2f})', 
            linewidth=1.5, linestyle='--', color='red')
    ax.plot(range(len(test_original)), manual_pred_original, 
            label=f'Manual (RMSE={manual_rmse:.2f})', 
            linewidth=1.5, linestyle=':', color='green')
    ax.set_title('Сравнение предсказаний на тестовой выборке')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Остатки manual модели
    ax = axes[2, 0]
    residuals = test_original - manual_pred_original
    ax.plot(residuals, color='purple', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=np.std(residuals)*2, color='orange', linestyle='--', alpha=0.5, label='±2σ')
    ax.axhline(y=-np.std(residuals)*2, color='orange', linestyle='--', alpha=0.5)
    ax.set_title(f'Остатки manual модели (σ={np.std(residuals):.2f})')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('Ошибка')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Прогноз на будущее
    ax = axes[2, 1]
    forecast_steps = 24
    
    # Прогноз manual модели
    manual_forecast_scaled = manual_model.predict(steps=forecast_steps)
    manual_forecast = scaler.inverse_transform(manual_forecast_scaled.reshape(-1, 1)).flatten()
    
    # Визуализация
    last_100 = pt08.values[-100:]
    ax.plot(range(100), last_100, label='Исторические данные', alpha=0.7, linewidth=1)
    forecast_x = range(100, 100 + forecast_steps)
    ax.plot(forecast_x, manual_forecast, 
            label=f'Прогноз ARMA({p},{q})', 
            color='red', linewidth=2, marker='o', markersize=4)
    ax.axvline(x=100, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title(f'Прогноз на {forecast_steps} шагов вперед')
    ax.set_xlabel('Временные шаги (от последних 100)')
    ax.set_ylabel('PT08.S3(NOx)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'ARMA({p},{q}) модель для предсказания временного ряда', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Прогноз на более длительный период
    print("\n" + "="*60)
    print(f"ПРОГНОЗ НА {forecast_steps} ШАГОВ ВПЕРЕД:")
    print("="*60)
    print("Шаг\tПредсказание\tДов. интервал")
    print("-" * 40)
    
    for i, val in enumerate(manual_forecast):
        ci = 1.96 * manual_model.sigma * scaler.scale_  # 95% доверительный интервал
        print(f"{i+1}\t{val:.2f}\t\t±{ci[0]:.2f}")
    
    # Сравнение с простой моделью (для контекста)
    print("\n" + "="*60)
    print("СРАВНЕНИЕ С ПРОСТЫМИ МОДЕЛЯМИ:")
    print("="*60)
    
    # Наивный прогноз (последнее значение)
    naive_pred = np.ones_like(test_original) * train_original[-1]
    naive_rmse = np.sqrt(np.mean((naive_pred - test_original) ** 2))
    
    # Среднее значение
    mean_pred = np.ones_like(test_original) * np.mean(train_original)
    mean_rmse = np.sqrt(np.mean((mean_pred - test_original) ** 2))
    
    print(f"Наивная модель (последнее значение): RMSE = {naive_rmse:.2f}")
    print(f"Среднее значение: RMSE = {mean_rmse:.2f}")
    print(f"ARMA модель (statsmodels): RMSE = {sm_rmse:.2f}")
    print(f"ARMA модель (manual): RMSE = {manual_rmse:.2f}")
    
    # Сохраняем коэффициенты для задания
    print("\n" + "="*60)
    print("КОЭФФИЦИЕНТЫ МОДЕЛИ ДЛЯ ГЕНЕРАЦИИ РЯДА:")
    print("="*60)
    print(f"p = {p}")
    print(f"q = {q}")
    print(f"c = {manual_model.c:.4f}")
    print(f"φ (AR) = {manual_model.phi}")
    print(f"θ (MA) = {manual_model.theta}")
    print(f"σ = {manual_model.sigma:.4f}")
    
    return {
        'p': p,
        'q': q,
        'manual_model': manual_model,
        'sm_model': sm_fitted,
        'scaler': scaler,
        'forecast': manual_forecast,
        'metrics': {
            'sm_rmse': sm_rmse,
            'manual_rmse': manual_rmse,
            'naive_rmse': naive_rmse,
            'mean_rmse': mean_rmse
        }
    }


def generate_series_from_model(p, q, phi, theta, c, sigma, n_points=1000):
    """
    Генерация временного ряда по заданным коэффициентам ARMA модели
    X_t = c + ε_t + Σ(φ_i * X_{t-i}) + Σ(θ_j * ε_{t-j})
    """
    # Инициализация
    n_burn = 200  # для стабилизации
    total_points = n_points + n_burn
    
    series = np.zeros(total_points)
    eps = np.random.normal(0, sigma, total_points)
    
    # Генерация ряда
    for t in range(max(p, q), total_points):
        # AR часть
        ar_part = c
        for i in range(1, p + 1):
            if i <= len(phi):
                ar_part += phi[i-1] * series[t-i]
        
        # MA часть
        ma_part = 0
        for j in range(1, q + 1):
            if j <= len(theta):
                ma_part += theta[j-1] * eps[t-j]
        
        series[t] = ar_part + ma_part + eps[t]
    
    # Убираем начальные значения
    return series[n_burn:]


if __name__ == "__main__":
    # Загрузка данных
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], 
                                      format='%d/%m/%Y %H.%M.%S')
    
    # Запуск ARMA предсказания
    results = arma_prediction(data)
    
    # Дополнительно: генерация нового ряда по коэффициентам обученной модели
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ НОВОГО РЯДА ПО КОЭФФИЦИЕНТАМ МОДЕЛИ")
    print("="*60)
    
    generated = generate_series_from_model(
        p=results['p'],
        q=results['q'],
        phi=results['manual_model'].phi,
        theta=results['manual_model'].theta,
        c=results['manual_model'].c,
        sigma=results['manual_model'].sigma,
        n_points=500
    )
    
    # Визуализация сгенерированного ряда
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(generated, color='purple', linewidth=0.8)
    ax.set_title(f'Сгенерированный ряд по ARMA({results["p"]},{results["q"]}) модели')
    ax.set_xlabel('Временные шаги')
    ax.set_ylabel('Значение')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Сгенерировано {len(generated)} точек")
    print(f"Статистика сгенерированного ряда:")
    print(f"  Среднее: {generated.mean():.4f}")
    print(f"  Std: {generated.std():.4f}")