import pandas as pd

import seaborn as sns
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
    correlation_matrix = data.corr()
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt='.2f')
    plt.show()

# 2. Вывести среднее значение изучаемой величины и дисперсию
def point_2(data):
    mean_value = data.mean()
    variance_values = data.var()

    print("Среднее значение:")
    print(mean_value)
    print("Дисперсия:")
    print(variance_values)

# 3. Изменяется ли диапазон доступных значений с изменением 
# временного периода или другой подвергаемой анализу характеристики?
def point_3(data):
    pass

if __name__== "__main__":
    # dax - Германия, smi - Швейцария, cac - Франция, ftse - Великобритания
    data = pd.read_csv("data/EuStockMarkets.csv")
    graph(data)
    point_3(data)