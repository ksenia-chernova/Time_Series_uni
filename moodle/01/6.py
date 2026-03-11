# Python: Пример ложной корреляции и теста на коинтеграцию (синтетические данные)
from statsmodels.tsa.stattools import coint
import numpy as np

# Создадим два независимых случайных блуждания (нестационарных)
np.random.seed(123)
X = np.random.randn(1000).cumsum()
Y = np.random.randn(1000).cumsum()

# 1. Корреляция исходных рядов
corr_spurious = np.corrcoef(X, Y)[0, 1]
print(f"Корреляция двух независимых случайных блужданий: {corr_spurious:.3f}")
# (Она может быть высокой, например, 0.5 или 0.6, хотя связи нет)

# 2. Корреляция разностей (стационарных рядов)
X_diff = np.diff(X)
Y_diff = np.diff(Y)
corr_diff = np.corrcoef(X_diff, Y_diff)[0, 1]
print(f"Корреляция разностей: {corr_diff:.3f}")  # Должна быть близка к 0

# 3. Тест на коинтеграцию (Энгла-Грейнджера)
# H0: ряды не коинтегрированы.
score, pvalue, _ = coint(X, Y)
print(f"p-value теста на коинтеграцию: {pvalue:.3f}")
# Если p-value > 0.05, мы не отвергаем H0 и делаем вывод об отсутствии коинтеграции.