"""
Задание 2: Построение АЧХ и ФЧХ для сигналов тремора
Чтение данных из файла lab5_data6.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


def read_tremor_data(filename='data\lab5_data6.txt'):
    """
    Чтение данных тремора из файла
    
    Параметры:
        filename (str): путь к файлу с данными
    
    Возвращает:
        samples (ndarray): номера отсчётов
        channel0 (ndarray): данные канала 0
        channel1 (ndarray): данные канала 1
        sampling_rate (float): частота дискретизации (Гц)
    """
    samples = []
    channel0 = []
    channel1 = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # пропускаем заголовок
        
        for row in reader:
            if len(row) >= 3 and row[0].strip():
                try:
                    samples.append(int(row[0]))
                    channel0.append(float(row[1].replace(',', '.')))
                    channel1.append(float(row[2].replace(',', '.')))
                except (ValueError, IndexError):
                    continue
    
    samples = np.array(samples)
    channel0 = np.array(channel0)
    channel1 = np.array(channel1)
    
    # Частота дискретизации для тремора (типичное значение 100 Гц)
    # При необходимости измените на реальное значение
    sampling_rate = 100.0
    
    return samples, channel0, channel1, sampling_rate


def dft(x):
    """
    Прямое дискретное преобразование Фурье (ручная реализация)
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    n = np.arange(N)
    
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * n / N))
    
    return X


def compute_frequency_response(signal, sampling_rate):
    """
    Вычисление АЧХ и ФЧХ сигнала
    
    Параметры:
        signal (ndarray): входной сигнал
        sampling_rate (float): частота дискретизации
    
    Возвращает:
        freqs (ndarray): массив частот
        magnitude (ndarray): АЧХ
        phase (ndarray): ФЧХ (в градусах)
    """
    # Убираем постоянную составляющую
    signal_detrended = signal - np.mean(signal)
    
    # Оконная функция Ханна для уменьшения утечки спектра
    window = np.hanning(len(signal_detrended))
    signal_windowed = signal_detrended * window
    
    # Прямое ДПФ
    X = dft(signal_windowed)
    
    # Частотная ось
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    
    # АЧХ (нормированная)
    magnitude = np.abs(X) / len(signal)
    magnitude[1:] *= 2  # Корректировка для одностороннего спектра
    
    # ФЧХ (в градусах)
    phase = np.angle(X, deg=True)
    
    return freqs, magnitude, phase


def plot_signal_and_spectrum(samples, signal, sampling_rate, channel_name, max_freq=30):
    """
    Построение графиков сигнала, АЧХ и ФЧХ
    """
    time = samples / sampling_rate
    
    # Вычисление спектра
    freqs, magnitude, phase = compute_frequency_response(signal, sampling_rate)
    half_idx = len(freqs) // 2
    display_idx = np.where((freqs[:half_idx] >= 0) & (freqs[:half_idx] <= max_freq))[0]
    
    # Создание фигуры с 3 подграфиками
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Сигнал во временной области
    axes[0].plot(time, signal, 'b-', linewidth=1)
    axes[0].set_xlabel('Время (с)')
    axes[0].set_ylabel('Амплитуда')
    axes[0].set_title(f'Сигнал тремора - {channel_name}')
    axes[0].grid(True, alpha=0.3)
    
    # 2. АЧХ
    axes[1].plot(freqs[display_idx], magnitude[display_idx], 'b-', linewidth=1.5)
    axes[1].set_xlabel('Частота (Гц)')
    axes[1].set_ylabel('Амплитуда')
    axes[1].set_title(f'АЧХ - {channel_name}')
    axes[1].grid(True, alpha=0.3)
    
    # Отмечаем пики на АЧХ
    peak_threshold = 0.15 * np.max(magnitude[display_idx])
    for i in range(2, len(display_idx) - 2):
        if (magnitude[display_idx[i]] > magnitude[display_idx[i-1]] and 
            magnitude[display_idx[i]] > magnitude[display_idx[i+1]] and
            magnitude[display_idx[i]] > peak_threshold):
            peak_freq = freqs[display_idx[i]]
            axes[1].axvline(x=peak_freq, color='r', linestyle='--', alpha=0.4)
            axes[1].text(peak_freq, magnitude[display_idx[i]] * 1.1, 
                        f'{peak_freq:.1f} Гц', fontsize=9, color='r', ha='center')
    
    # 3. ФЧХ
    axes[2].plot(freqs[display_idx], phase[display_idx], 'g-', linewidth=1)
    axes[2].set_xlabel('Частота (Гц)')
    axes[2].set_ylabel('Фаза (градусы)')
    axes[2].set_title(f'ФЧХ - {channel_name}')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-180, 180)
    axes[2].set_yticks([-180, -90, 0, 90, 180])
    
    plt.tight_layout()
    return fig, freqs, magnitude, phase


def print_spectrum_peaks(freqs, magnitude, channel_name, max_freq=30, top_n=5):
    """
    Вывод основных частотных пиков
    """
    half_idx = len(freqs) // 2
    display_idx = np.where((freqs[:half_idx] >= 0) & (freqs[:half_idx] <= max_freq))[0]
    
    # Поиск пиков
    peaks = []
    for i in range(2, len(display_idx) - 2):
        if (magnitude[display_idx[i]] > magnitude[display_idx[i-1]] and 
            magnitude[display_idx[i]] > magnitude[display_idx[i+1]]):
            peaks.append((freqs[display_idx[i]], magnitude[display_idx[i]]))
    
    # Сортировка по амплитуде
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nОсновные частотные компоненты - {channel_name}:")
    print("  Частота (Гц) | Амплитуда")
    print("  -------------|-----------")
    for freq, amp in peaks[:top_n]:
        print(f"  {freq:11.2f} | {amp:.6f}")
    
    return peaks


def main():
    """Основная функция"""
    
    print("=" * 60)
    print("ЗАДАНИЕ 2: АНАЛИЗ АЧХ И ФЧХ СИГНАЛОВ ТРЕМОРА")
    print("=" * 60)
    
    # Загрузка данных
    filename = 'data\lab5_data6.txt'
    print(f"\nЧтение данных из файла: {filename}")
    
    try:
        samples, channel0, channel1, sampling_rate = read_tremor_data(filename)
        print(f"  Загружено {len(samples)} отсчётов")
        print(f"  Частота дискретизации: {sampling_rate} Гц")
        print(f"  Длительность записи: {len(samples)/sampling_rate:.2f} с")
    except FileNotFoundError:
        print(f"  ОШИБКА: Файл {filename} не найден!")
        print("  Убедитесь, что файл находится в текущей директории.")
        return
    
    # Статистика сигналов
    print(f"\nСтатистика сигналов:")
    print(f"  Канал 0: среднее = {np.mean(channel0):.6f}, СКО = {np.std(channel0):.6f}")
    print(f"  Канал 1: среднее = {np.mean(channel1):.6f}, СКО = {np.std(channel1):.6f}")
    
    # Анализ Канала 0
    print("\n" + "-" * 40)
    print("АНАЛИЗ КАНАЛА 0")
    print("-" * 40)
    
    fig0, freqs, mag0, phase0 = plot_signal_and_spectrum(
        samples, channel0, sampling_rate, "Канал 0", max_freq=30
    )
    peaks0 = print_spectrum_peaks(freqs, mag0, "Канал 0", max_freq=30)
    
    # Анализ Канала 1
    print("\n" + "-" * 40)
    print("АНАЛИЗ КАНАЛА 1")
    print("-" * 40)
    
    fig1, freqs, mag1, phase1 = plot_signal_and_spectrum(
        samples, channel1, sampling_rate, "Канал 1", max_freq=30
    )
    peaks1 = print_spectrum_peaks(freqs, mag1, "Канал 1", max_freq=30)
    
    # Сравнение спектров каналов
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СПЕКТРОВ КАНАЛОВ")
    print("=" * 60)
    
    plt.figure(figsize=(14, 5))
    
    half_idx = len(freqs) // 2
    max_freq = 30
    display_idx = np.where((freqs[:half_idx] >= 0) & (freqs[:half_idx] <= max_freq))[0]
    
    plt.plot(freqs[display_idx], mag0[display_idx], 'b-', label='Канал 0', alpha=0.7)
    plt.plot(freqs[display_idx], mag1[display_idx], 'r-', label='Канал 1', alpha=0.7)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title('Сравнение АЧХ каналов')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Вывод заключения
    print("\n" + "=" * 60)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 60)
    
    if len(peaks0) > 0 and len(peaks1) > 0:
        main_freq0 = peaks0[0][0] if peaks0 else 0
        main_freq1 = peaks1[0][0] if peaks1 else 0
        print(f"Доминирующая частота тремора (Канал 0): {main_freq0:.2f} Гц")
        print(f"Доминирующая частота тремора (Канал 1): {main_freq1:.2f} Гц")
        
        if abs(main_freq0 - main_freq1) < 0.5:
            print("\nЧастотные характеристики каналов хорошо коррелируют.")
        else:
            print("\nНаблюдаются различия в частотных характеристиках каналов.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()