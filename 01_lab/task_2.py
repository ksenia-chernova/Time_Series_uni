import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg


def stationary_row(data):
    pt08 = data['PT08.S3(NOx)'].dropna()

    result = adfuller(pt08)
    print('ADF Statistic (PT08.S3(NOx)): %f' % result[0])
    print('p-value (PT08.S3(NOx)): %f' % result[1])
    print('Critical Values (PT08.S3(NOx)):')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def graph(data):
    data = data.sort_values('Datetime')

    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()

    xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.xticks(rotation=90)
    plt.plot(data['Datetime'], data['PT08.S3(NOx)'])
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("data/AirQualityUCI.csv", sep=';')
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')
    stationary_row(data)
    graph(data)
