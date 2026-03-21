import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lab3_data6.txt", sep="\t")
df.columns = ['Nothing', 'Aggressive', 'White noise', 'Classical', 'Rhythmic']
df.insert(0, 'Interval', range(len(df)))

fig, axes = plt.subplots(5, 1, figsize=(10, 12))

axes[0].plot(df['Interval'], df['Nothing'], color='red', label='Nothing')
axes[0].set_ylabel('Heart rate')
axes[0].legend()
axes[0].grid()

axes[1].plot(df['Interval'], df['Aggressive'], color='green', label='Aggressive')
axes[1].set_ylabel('Heart rate')
axes[1].legend()
axes[1].grid()

axes[2].plot(df['Interval'], df['White noise'], color='blue', label='White noise')
axes[2].set_ylabel('Heart rate')
axes[2].legend()
axes[2].grid()

axes[3].plot(df['Interval'], df['Classical'], color='orange', label='Classical')
axes[3].set_ylabel('Heart rate')
axes[3].legend()
axes[3].grid()

axes[4].plot(df['Interval'], df['Rhythmic'], color='black', label='Rhythmic')
axes[4].set_xlabel('Interval')
axes[4].set_ylabel('Heart rate')
axes[4].legend()
axes[4].grid()

plt.tight_layout()
plt.show()