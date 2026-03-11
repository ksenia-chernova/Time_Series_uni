# Python: Расширенный тест Дики-Фуллера (ADF) на данных EuStockMarkets
from statsmodels.tsa.stattools import adfuller
from statsmodels.datasets import get_rdataset

data = get_rdataset('EuStockMarkets').data
dax = data['DAX'].dropna()

# Тест для исходного ряда цен (нестационарного)
result_prices = adfuller(dax)
print('ADF Statistic (Prices): %f' % result_prices[0])
print('p-value (Prices): %f' % result_prices[1])
print('Critical Values (Prices):')
for key, value in result_prices[4].items():
    print('\t%s: %.3f' % (key, value))

print('-' * 40)
# Тест для ряда изменений (разностей) - должен быть стационарным
dax_returns = dax.diff().dropna()
result_returns = adfuller(dax_returns)
print('ADF Statistic (Returns): %f' % result_returns[0])
print('p-value (Returns): %f' % result_returns[1])
print('Critical Values (Returns):')
for key, value in result_returns[4].items():
    print('\t%s: %.3f' % (key, value))

# Ожидаемый результат: для цен p-value будет большим (>0.05), для изменений - очень маленьким (<0.05).