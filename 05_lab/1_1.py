# ручная

import numpy as np
import matplotlib.pyplot as plt


def dft(x):
    """
    Прямое дискретное преобразование Фурье (ручная реализация)
    
    Параметры:
        x (ndarray): входной сигнал (одномерный массив)
    
    Возвращает:
        X (ndarray): комплексный спектр сигнала
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    n = np.arange(N)
    
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * n / N))
    
    return X


def idft(X):
    """
    Обратное дискретное преобразование Фурье (ручная реализация)
    
    Параметры:
        X (ndarray): комплексный спектр сигнала
    
    Возвращает:
        x (ndarray): восстановленный сигнал (вещественная часть)
    """
    N = len(X)
    x = np.zeros(N, dtype=complex)
    k = np.arange(N)
    
    for n in range(N):
        x[n] = (1 / N) * np.sum(X * np.exp(2j * np.pi * k * n / N))
    
    return x.real


def generate_signal(A0, omega0, phi0, duration, sampling_rate):
    """
    A0: амплитуда
    omega0: угловая частота (рад/с)
    phi0: начальная фаза (рад)
    duration: длительность сигнала (с)
    sampling_rate: частота дискретизации (Гц)

    x: массив временных отсчётов
    y: массив значений сигнала
    """

    num_samples = int(sampling_rate * duration)
    x = np.linspace(0, duration, num_samples, endpoint=False)
    y = A0 * np.sin(omega0 * x + phi0)
    return x, y


def plot_signal(x, y, title="Сигнал"):
    """Построение графика сигнала во временной области"""

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, 'b-', linewidth=1.5)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_spectrum(freqs, amplitude, title="Амплитудный спектр", max_freq=None):
    """Построение амплитудного спектра"""
    plt.figure(figsize=(12, 4))
    
    # Берём только положительные частоты (первую половину)
    half_idx = len(freqs) // 2
    plt.stem(freqs[:half_idx], amplitude[:half_idx], basefmt=" ")
    
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if max_freq is not None:
        plt.xlim(0, max_freq)
    
    plt.tight_layout()
    return plt.gcf()


def main():
    
    # Параметры сигнала
    A0 = 1.0                     # амплитуда
    f0 = 2.0                     # частота (Гц)
    omega0 = 5.0 * np.pi * f0    # угловая частота
    phi0 = np.pi / 4             # начальная фаза (45 градусов)
    duration = 1.0               # длительность (с)
    sampling_rate = 100.0        # частота дискретизации (Гц)

    print(f"Параметры сигнала:")
    print(f"  Амплитуда: {A0}")
    print(f"  Частота: {f0} Гц")
    print(f"  Фаза: {phi0:.3f} рад")
    print(f"  Длительность: {duration} с")
    print(f"  Частота дискретизации: {sampling_rate} Гц")
    print(f"  Количество отсчётов: {int(sampling_rate * duration)}")
    
    # 1. Генерация исходного сигнала
    x, y = generate_signal(A0, omega0, phi0, duration, sampling_rate)
    
    print("\n1. Исходный сигнал сгенерирован")
    fig1 = plot_signal(x, y, "Исходный сигнал: синусоида")
    
    # 2. Прямое ДПФ
    print("\n2. Выполнение прямого ДПФ (ручная реализация)...")
    X = dft(y)
    
    # Расчёт частотной оси
    freqs = np.fft.fftfreq(len(x), 1 / sampling_rate)
    
    # Нормировка амплитудного спектра
    amplitude = np.abs(X) / len(x)
    amplitude[1:] *= 2  # Корректировка для одностороннего спектра
    
    print("   Прямое ДПФ выполнено")
    fig2 = plot_spectrum(freqs, amplitude, 
                         "Амплитудный спектр (ручное ДПФ)", 
                         max_freq=10)
    
    # Поиск пика в спектре
    half_idx = len(freqs) // 2
    peak_idx = np.argmax(amplitude[:half_idx])
    print(f"   Пик спектра: частота = {freqs[peak_idx]:.3f} Гц, "
          f"амплитуда = {amplitude[peak_idx]:.3f}")
    
    # 3. Обратное ДПФ
    print("\n3. Выполнение обратного ДПФ (ручная реализация)...")
    y_reconstructed = idft(X)
    print("   Обратное ДПФ выполнено")
    
    # 4. Сравнение исходного и восстановленного сигналов
    mse = np.mean((y - y_reconstructed) ** 2)
    max_diff = np.max(np.abs(y - y_reconstructed))
    
    print("\n4. Результаты восстановления:")
    print(f"   Среднеквадратичная ошибка (MSE): {mse:.2e}")
    print(f"   Максимальная абсолютная разница: {max_diff:.2e}")
    
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, 'b-', label='Исходный сигнал', linewidth=2, alpha=0.7)
    plt.plot(x, y_reconstructed, 'r--', label='Восстановленный сигнал', linewidth=2)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Сравнение: исходный и восстановленный сигнал (ручное ДПФ)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()