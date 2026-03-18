import pandas as pd
import numpy as np
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

# 6. Построить две диаграммы рассеяния: для определения 
# взаимосвязи между ценами двух акций в отдельные моменты времени и для отслеживания их временных изменений.
def point_6(data):
    stock1 = 'DAX'
    stock2 = 'SMI'
    data['period'] = pd.cut(data.index, bins=3, labels=['Начало', 'Середина', 'Конец'])


    # plt.figure(figsize=(10, 8))
    # plt.scatter(data[stock1], data[stock2], alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    z = np.polyfit(data[stock1], data[stock2], 1)
    p = np.poly1d(z)
    # plt.plot(data[stock1], p(data[stock1]), "red", linewidth=2, label=f'Линия тренда')
    correlation = data[stock1].corr(data[stock2])
    # plt.title(f'Рис. 1: Взаимосвязь между {stock1} и {stock2}\nКорреляция: {correlation:.3f}', 
    #         fontsize=14, fontweight='bold')
    # plt.xlabel(f'{stock1} (пункты)', fontsize=12)
    # plt.ylabel(f'{stock2} (пункты)', fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()


    plt.figure(figsize=(10, 8))
    colors = {'Начало': 'blue', 'Середина': 'green', 'Конец': 'red'}
    for period, color in colors.items():
        period_data = data[data['period'] == period]
        plt.scatter(period_data[stock1], period_data[stock2], 
                c=color, label=period, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    for period, color in colors.items():
        period_data = data[data['period'] == period]
        if len(period_data) > 1:
            z = np.polyfit(period_data[stock1], period_data[stock2], 1)
            p = np.poly1d(z)
            plt.plot(period_data[stock1], p(period_data[stock1]), 
                    color=color, linewidth=2, linestyle='--')
    plt.title(f'Рис. 2: Изменение взаимосвязи {stock1} и {stock2} во времени', 
            fontsize=14, fontweight='bold')
    plt.xlabel(f'{stock1} (пункты)', fontsize=12)
    plt.ylabel(f'{stock2} (пункты)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nОбщая корреляция между {stock1} и {stock2}: {correlation:.3f}")

    print("\nКорреляция по периодам:")
    for period in ['Начало', 'Середина', 'Конец']:
        period_data = data[data['period'] == period]
        corr = period_data[stock1].corr(period_data[stock2])
        print(f"  {period}: {corr:.3f}")

def point_7(data):
    numeric_data = data.select_dtypes(include=[np.number])
    cov = numeric_data.cov()
    print("Ковариационная матрица:")
    print(cov)

"""
8. Прочитать про ложную корреляцию стр. 135 - 137. Найти и продемонстрировать интересную ложную корреляцию на упомянутом сайте.
https://tylervigen.com/spurious-correlations
https://tylervigen.com/spurious/correlation/5139_popularity-of-the-first-name-alaina_correlates-with_popularity-of-the-trollface-meme
"""

if __name__== "__main__":
    # dax - Германия, smi - Швейцария, cac - Франция, ftse - Великобритания
    data = pd.read_csv("data/EuStockMarkets.csv")
    graph(data)
    del data['rownames']
    point_1(data)
    point_2(data)
    point_5(data)
    point_6(data)
    point_7(data)
    