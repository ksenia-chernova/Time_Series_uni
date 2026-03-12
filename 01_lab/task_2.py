import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv("data/AirQualityUCI.csv", sep=';')

data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')

data = data.sort_values('Datetime')

fig, ax = plt.subplots(1)
fig.autofmt_xdate()

plt.plot(data['Datetime'], range(len(data)))

xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
ax.xaxis.set_major_formatter(xfmt)

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()