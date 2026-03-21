import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock
from fastdtw import fastdtw  # установите: pip install fastdtw

# Генерация трёх временных рядов разной длины и формы
np.random.seed(42)
t = np.linspace(0, 4*np.pi, 50)

# Ряд A: чистая синусоида
series_a = np.sin(t)

# Ряд B: синусоида со сдвигом по фазе и небольшим шумом
series_b = np.sin(t + 1.0) + 0.1 * np.random.randn(len(t))

# Ряд C: синусоида с другим масштабом, меньшей длиной и шумом
t_short = np.linspace(0, 4*np.pi, 40)
series_c = 0.8 * np.sin(t_short * 1.2) + 0.2 * np.random.randn(len(t_short))

# Визуализация
plt.figure(figsize=(10, 4))
plt.plot(t, series_a, label='Ряд A (синус)')
plt.plot(t, series_b, label='Ряд B (сдвиг+шум)')
plt.plot(t_short, series_c, label='Ряд C (короткий, масштаб)')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.title('Примеры временных рядов')
plt.show()

# Функция для печати расстояний
def print_distances(a, b, name_a, name_b):
    # Евклидово (требует одинаковой длины)
    if len(a) == len(b):
        ed = euclidean(a, b)
        print(f'{name_a} – {name_b}: евклидово = {ed:.4f}')
    else:
        print(f'{name_a} – {name_b}: евклидово — разная длина (нельзя вычислить)')

    # Манхэттенское
    if len(a) == len(b):
        md = cityblock(a, b)
        print(f'{name_a} – {name_b}: манхэттенское = {md:.4f}')
    else:
        print(f'{name_a} – {name_b}: манхэттенское — разная длина')

    # DTW (работает с разной длиной)
    distance, path = fastdtw(a, b)
    print(f'{name_a} – {name_b}: DTW = {distance:.4f}')
    print()

# Попарные сравнения
print_distances(series_a, series_b, 'A', 'B')
print_distances(series_a, series_c, 'A', 'C')
print_distances(series_b, series_c, 'B', 'C')