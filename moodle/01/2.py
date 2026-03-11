# Python: Корреляция цен и корреляция изменений со смещением (лагом) на данных EuStockMarkets
import pandas as pd
import numpy as np
from statsmodels.datasets import get_rdataset

data = get_rdataset('EuStockMarkets').data
df = data[['DAX', 'SMI']].copy()
df.columns = ['DAX', 'SMI']

# 1. Корреляция исходных цен (как на левом графике)
corr_prices = df['DAX'].corr(df['SMI'])
print(f"Корреляция цен DAX и SMI: {corr_prices:.3f}")  # Будет очень высокой (~0.99)

# 2. Корреляция ежедневных изменений (как на правом графике)
returns_dax = df['DAX'].diff().dropna()
returns_smi = df['SMI'].diff().dropna()
corr_returns = returns_dax.corr(returns_smi)
print(f"Корреляция изменений DAX и SMI: {corr_returns:.3f}")  # Значительная, но меньше

# 3. Корреляция со смещением (лагом) - попытка предсказать одно другим
# Смещаем изменения DAX на 1 день назад (lag)
returns_dax_lagged = returns_dax.shift(1).dropna()
# Выравниваем индексы для корректного расчета
aligned_data = pd.concat([returns_dax_lagged, returns_smi], axis=1, join='inner').dropna()
corr_lagged = aligned_data.iloc[:,0].corr(aligned_data.iloc[:,1])
print(f"Корреляция вчерашних изменений DAX с сегодняшними изменениями SMI: {corr_lagged:.3f}")
# Вывод: корреляция, скорее всего, будет близка к нулю, что указывает на отсутствие простой предсказательной способности.