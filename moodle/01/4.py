# Python: Построение ACF и PACF на синтетическом примере (синусоида) как в книге
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Пример с чистой синусоидой (как в книге)
x = np.arange(1, 101)
y = np.sin(x * np.pi / 3)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(y, marker='o', linestyle='-')
plt.title('Ряд: sin(x * pi/3)')

plt.subplot(2, 2, 2)
plot_acf(y, lags=30, ax=plt.gca(), zero=False)  # zero=False чтобы не показывать лаг 0
plt.title('ACF')

plt.subplot(2, 2, 3)
plot_pacf(y, lags=30, ax=plt.gca(), zero=False, method='ywm')
plt.title('PACF')
plt.tight_layout()
plt.show()

# Вывод: На ACF мы видим периодичность (корреляция на лагах 6, 12, 18...),
# На PACF значима только корреляция на лаге 6, т.к. она "очищена" от влияния промежуточных точек.