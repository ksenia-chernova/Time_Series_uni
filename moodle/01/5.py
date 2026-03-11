# Python: ACF и PACF для данных AirPassengers из statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd

# Загружаем данные AirPassengers
air = sm.datasets.get_rdataset("AirPassengers").data['value']
air.index = pd.date_range(start='1949-01', periods=len(air), freq='ME')

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
air.plot()
plt.title('Исходный ряд (AirPassengers)')

plt.subplot(2, 2, 2)
plot_acf(air, lags=40, ax=plt.gca())
plt.title('ACF')

plt.subplot(2, 2, 3)
plot_pacf(air, lags=40, ax=plt.gca(), method='ywm')
plt.title('PACF')
plt.tight_layout()
plt.show()

# Интерпретация:
# - ACF медленно убывает и имеет ярко выраженные пики на лагах, кратных 12, что указывает на сильный тренд и сезонность.
# - PACF также показывает значимые пики на лагах 1, 12, 24, но они слабее, чем в ACF.