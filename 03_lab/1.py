import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/lab3_data6.txt", sep="\t")
df.columns = ['Nothing', 'Aggressive', 'White noise', 'Classical', 'Rhythmic']

fs = 20
dt = 1/fs

def rr_to_uniform_series(rr_intervals, dt, fs):
    rr_seconds = np.array(rr_intervals) / 1000.0
    beat_times = np.cumsum(rr_seconds)
    total_duration = beat_times[-1]
    n_samples = int(np.ceil(total_duration / dt)) + 1
    time_axis = np.arange(n_samples) * dt
    uniform_series = np.zeros(n_samples)
    
    for beat_time in beat_times:
        idx = int(np.round(beat_time / dt))
        if idx < n_samples:
            uniform_series[idx] = 1
    
    return uniform_series, time_axis

fig, axes = plt.subplots(5, 1, figsize=(15, 12))

titles = ['Nothing', 'Aggressive', 'White noise', 'Classical', 'Rhythmic']
colors = ['red', 'green', 'blue', 'orange', 'black']

for idx, (title, color) in enumerate(zip(titles, colors)):
    rr_intervals = df[title].dropna().values
    uniform_series, time_axis = rr_to_uniform_series(rr_intervals, dt, fs)
    
    time_limit = 10
    mask = time_axis <= time_limit
    
    beat_times = time_axis[uniform_series == 1]
    for beat_time in beat_times[beat_times <= time_limit]:
        axes[idx].axvline(x=beat_time, color=color, alpha=0.5, linewidth=0.5)
    
    axes[idx].plot(time_axis[mask], uniform_series[mask], 
                   color=color, linewidth=1.5)
    
    axes[idx].set_ylabel('Сигнал ударов')
    axes[idx].set_xlabel('Время (с)')
    axes[idx].set_title(f'{title}')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim(-0.1, 1.1)
    axes[idx].set_xlim(0, time_limit)

plt.tight_layout()
plt.show()