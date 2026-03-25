import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def taylor_coeffs_binomial(degree, n_max, point=0):
    coeffs = []
    for n in range(n_max + 1):
        if n == 0:
            coeff = (1 + point) ** degree
        else:
            num = 1.0
            for k in range(n):
                num *= (degree - k)
            f = factorial(n)
            coeff = (num / f) * ((1 + point) ** (degree - n))
        
        coeffs.append(coeff)
    print(coeffs)
    return coeffs

def taylor_series_at_point(x_vals, coeffs, point=0):
    result = np.zeros_like(x_vals, dtype=float)
    for n, c in enumerate(coeffs):
        result += c * ((x_vals - point) ** n)
    return result

degree = 0.5
expansion_point = 0
x_min, x_max = -0.9, 0.9
n_vals = [3, 5, 10, 25, 50]

x = np.linspace(x_min, x_max, 500)
y_exact = (1 + x) ** degree

plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, 'k-', linewidth=2, 
         label=f'Функция $(1+x)^{{{degree}}}$')

colors = plt.cm.viridis(np.linspace(0, 1, len(n_vals)))

for N, color in zip(n_vals, colors):
    coeffs = taylor_coeffs_binomial(degree, N-1, expansion_point)
    y_approx = taylor_series_at_point(x, coeffs, expansion_point)
    plt.plot(x, y_approx, '--', color=color, 
             label=f'{N} членов')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.xlim(x_min, x_max)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.show()