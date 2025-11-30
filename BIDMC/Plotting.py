import csv
import numpy as np

AAMI_key = {'N','r','S','V'}
AAMI = dict()

def Tolist(x):
    return list(map(float,x))

for key in AAMI_key:
    with open(f'{key}_BIDMC.csv', 'r') as f:
        reader = csv.reader(f)
        AAMI[key] = list(map(Tolist,list(reader)))

for i in AAMI_key:
    print(f'{i}={np.array(AAMI[i]).shape}')

import matplotlib.pyplot as plt

for category in AAMI_key:
    # Choose a category to plot, for example 'N'
    #category = 'N'
    ecg_data = AAMI[category]

    for i in range(1, 10):
        # Plot the first ECG signal from this category
        plt.plot(ecg_data[i])
        plt.title(f'ECG Signal - Category {category}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
