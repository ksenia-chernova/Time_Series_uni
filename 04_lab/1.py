import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def taylor_coeffs_binomial(n_max, point=0):
    coeffs = []
    x = sp.symbols('x')
    func = (1 + x) **0.5
    for n in range(n_max + 1):
        if n == 0:
            # f(a)
            coeff = float(func.subs(x, point))
        else:
            # расчёт nй производной
            f_n = sp.diff(func, x, n)
            # расчёт значения производной в точке 0
            f_n_at_point = f_n.subs(x, point)
            # коэффициент = f^(n)(point) / n!
            coeff = float(f_n_at_point  / sp.factorial(n))

        coeffs.append(coeff)

    print(coeffs)
    return coeffs

def taylor_series_at_point(x_vals, coeffs, point=0):
    result = np.zeros_like(x_vals, dtype=float)
    for n, coeff in enumerate(coeffs):
        # умножение коэффициента на (x - point)^n
        result += coeff * ((x_vals - point) ** n)
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
    coeffs = taylor_coeffs_binomial(N-1, expansion_point)
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