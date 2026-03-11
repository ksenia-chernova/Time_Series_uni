import pandas as pd
import matplotlib.pyplot as plt

# Графики изменения индексов
def graph(data):
    plt.plot(data['rownames'], data['DAX'], color='red', label='DAX')
    plt.plot(data['rownames'], data['SMI'], color='green', label='SMI')
    plt.plot(data['rownames'], data['CAC'], color='blue', label='CAC')
    plt.plot(data['rownames'], data['FTSE'], color='black', label='FTSE')

    plt.xlabel('num')
    plt.ylabel('indexes')
    plt.legend()
    plt.show()

# 1. Существуют ли в наборе данных взаимосвязанные столбцы?
def point_1(data):
    correlation_matrix_current_prices = data.corr()
    print("Матрица корреляции исходных цен:")
    print(correlation_matrix_current_prices)

    correlation_matrix_everyday_changes = data.diff().dropna().corr()
    print("Матрица корреляции ежедневных изменений:")
    print(correlation_matrix_everyday_changes)

# 2. Вывести среднее значение изучаемой величины и дисперсию
def point_2(data):
    mean_value = data.mean()
    variance_values = data.var()

    print("Среднее значение: \n", mean_value)
    print("Дисперсия: \n", variance_values)

# 5. Построить гистограмму абсолютных значений и гистограмму разностей. Сделать выводы.
def point_5(data):
    
    # абсолютные значения
    plt.figure(figsize=(12, 5))
    data.hist(bins=50, edgecolor='black')

    # разности
    returns = data.diff().dropna()
    plt.figure(figsize=(12, 5))
    returns.hist(bins=50, edgecolor='black')
    plt.show()

if __name__== "__main__":
    # dax - Германия, smi - Швейцария, cac - Франция, ftse - Великобритания
    data = pd.read_csv("data/EuStockMarkets.csv")
    graph(data)
    del data['rownames']
    # point_1(data)
    # point_2(data)
    point_5(data)
    