import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# dax - Германия, smi - Швейцария, cac - Франция, ftse - Великобритания
data = pd.read_csv("data/EuStockMarkets.csv", index_col=0)

# 1. Существуют ли в наборе данных взаимосвязанные столбцы?
def point_1():
    correlation_matrix = data.corr()
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt='.2f')
    plt.show()

# 2. Вывести среднее значение изучаемой величины и дисперсию
def point_2():
    mean_value = data.mean()
    variance_values = data.var()

    print("Среднее значение:")
    print(mean_value)
    print("Дисперсия:")
    print(variance_values)

if __name__=="__main__":
    point_2()