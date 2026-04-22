# numpy.fft

import numpy as np
import matplotlib.pyplot as plt


def generate_signal(A0, omega0, phi0, duration, sampling_rate):
    """
    A0 (float): амплитуда
    omega0 (float): угловая частота (рад/с)
    phi0 (float): начальная фаза (рад)
    duration (float): длительность сигнала (с)
    sampling_rate (float): частота дискретизации (Гц)

    t (ndarray): массив временных отсчётов
    y (ndarray): массив значений сигнала
    """

    num_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    y = A0 * np.sin(omega0 * t + phi0)
    return t, y


def compute_spectrum(y, sampling_rate):
    """
    y (ndarray): входной сигнал
    sampling_rate (float): частота дискретизации

    freqs (ndarray): массив частот
    X (ndarray): комплексный спектр
    amplitude (ndarray): нормированный амплитудный спектр
    """

    # Прямое БПФ
    X = np.fft.fft(y)
    
    # Частотная ось
    freqs = np.fft.fftfreq(len(y), 1 / sampling_rate)
    
    # Нормированный амплитудный спектр
    amplitude = np.abs(X) / len(y)
    amplitude[1:] *= 2  # Корректировка для одностороннего спектра
    
    return freqs, X, amplitude


def reconstruct_signal(X):
    """
    Восстановление сигнала с использованием numpy.fft.ifft
    
    Параметры:
        X (ndarray): комплексный спектр
    
    Возвращает:
        y_reconstructed (ndarray): восстановленный сигнал
    """
    return np.fft.ifft(X).real


def plot_signal(t, y, title="Сигнал"):
    """Построение графика сигнала во временной области"""
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, 'b-', linewidth=1.5)
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
    """Основная функция: демонстрация библиотечного БПФ"""
    
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
    t, y = generate_signal(A0, omega0, phi0, duration, sampling_rate)
    
    print("\n1. Исходный сигнал сгенерирован")
    fig1 = plot_signal(t, y, "Исходный сигнал: синусоида")
    
    # 2. Прямое БПФ (библиотечное)
    print("\n2. Выполнение прямого БПФ (numpy.fft)...")
    freqs, X, amplitude = compute_spectrum(y, sampling_rate)
    print("   Прямое БПФ выполнено")
    
    fig2 = plot_spectrum(freqs, amplitude, 
                         "Амплитудный спектр (numpy.fft)", 
                         max_freq=10)
    
    # Поиск пика в спектре
    half_idx = len(freqs) // 2
    peak_idx = np.argmax(amplitude[:half_idx])
    print(f"   Пик спектра: частота = {freqs[peak_idx]:.3f} Гц, "
          f"амплитуда = {amplitude[peak_idx]:.3f}")
    
    # 3. Обратное БПФ (библиотечное)
    print("\n3. Выполнение обратного БПФ (numpy.fft.ifft)...")
    y_reconstructed = reconstruct_signal(X)
    print("   Обратное БПФ выполнено")
    
    # 4. Сравнение исходного и восстановленного сигналов
    mse = np.mean((y - y_reconstructed) ** 2)
    max_diff = np.max(np.abs(y - y_reconstructed))
    
    print("\n4. Результаты восстановления:")
    print(f"   Среднеквадратичная ошибка (MSE): {mse:.2e}")
    print(f"   Максимальная абсолютная разница: {max_diff:.2e}")
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, 'b-', label='Исходный сигнал', linewidth=2, alpha=0.7)
    plt.plot(t, y_reconstructed, 'g--', label='Восстановленный сигнал', linewidth=2)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Сравнение: исходный и восстановленный сигнал (numpy.fft)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()