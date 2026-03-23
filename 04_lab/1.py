import numpy as np
import matplotlib.pyplot as plt

def taylor_coeffs_binomial(a, n_max):
    """
    Возвращает список коэффициентов Тейлора для (1+x)^a:
    coeffs[n] = f^(n)(0)/n! = C(a, n)
    """
    coeffs = [1.0]  # n=0
    for n in range(1, n_max + 1):
        # Рекуррентное соотношение для биномиальных коэффициентов
        # C(a, n) = C(a, n-1) * (a - n + 1) / n
        coeffs.append(coeffs[-1] * (a - n + 1) / n)
    return coeffs

def taylor_series(x_vals, coeffs):
    """
    Вычисляет значение ряда Тейлора в точках x_vals,
    используя коэффициенты coeffs[0..M].
    """
    result = np.zeros_like(x_vals, dtype=float)
    for n, c in enumerate(coeffs):
        result += c * (x_vals ** n)
    return result

# Параметры
a = 0.5          # (1+x)^0.5 = sqrt(1+x)
x_min, x_max = -0.9, 0.9  # интервал, где ряд сходится (|x| < 1)
n_vals = [3, 5, 10, 25, 50]  # количество членов ряда (включая нулевой)

# Точная функция
x = np.linspace(x_min, x_max, 500)
y_exact = (1 + x) ** a

# Строим графики
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, 'k-', linewidth=2, label='Точная функция $(1+x)^{0.5}$')

colors = plt.cm.viridis(np.linspace(0, 1, len(n_vals)))

for N, color in zip(n_vals, colors):
    # Получаем коэффициенты для N членов (от n=0 до N-1)
    coeffs = taylor_coeffs_binomial(a, N-1)
    y_approx = taylor_series(x, coeffs)
    plt.plot(x, y_approx, '--', color=color, label=f'{N} членов')

plt.title(f'Аппроксимация функции $(1+x)^{{{a}}}$ рядом Тейлора (точка 0)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.xlim(x_min, x_max)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.show()
