# Python: Исследовательский анализ с использованием датасетов из statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.datasets import get_rdataset

# Загружаем данные EuStockMarkets
data = get_rdataset('EuStockMarkets').data
# В оригинале это объект ts, в pandas преобразуем индексы во временную шкалу
# Для простоты используем числовой индекс
prices = data[['DAX', 'SMI', 'CAC', 'FTSE']]  # это DataFrame с колонками индексов

# 1. Визуализация исходного ряда (например, SMI)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
prices['SMI'].plot(title='Исходный ряд цен (SMI)')
plt.ylabel('Цена')

# 2. Гистограмма исходного ряда
plt.subplot(1, 2, 2)
prices['SMI'].hist(bins=50, edgecolor='black')
plt.title('Гистограмма исходного ряда SMI')
plt.tight_layout()
plt.show()

# 3. Анализ изменений (разностей) - гораздо более информативно
returns = prices['SMI'].diff().dropna()  # ежедневные изменения

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
returns.plot(title='Ежедневные изменения (разности) SMI')
plt.ylabel('Изменение цены')

plt.subplot(1, 2, 2)
returns.hist(bins=50, edgecolor='black')
plt.title('Гистограмма изменений SMI')
plt.tight_layout()
plt.show()

# Вывод: гистограмма изменений (часто похожая на нормальное распределение) 
# дает гораздо больше информации для анализа, чем гистограмма исходного трендового ряда.