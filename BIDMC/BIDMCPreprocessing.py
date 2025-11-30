import wfdb
import numpy as np
import csv
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Adapt this dictionary based on the European ST-T annotations
AAMI_BIDMC = {
    'N': 'N',  # Normal beat
    'S': 'S',  # Supraventricular premature or ectopic beat
    'V': 'V',  # Premature ventricular contraction
    'r': 'r'   # Fusion of ventricular and normal beat
}

# BIDMC File names
file_name = ['chf02', 'chf03', 'chf05', 'chf07', 'chf08', 'chf09', 'chf14']

for F_name in file_name:
    # read files
    sig, fields = wfdb.rdsamp(f'BIDMC/{F_name}')

    signal_index = 1 if F_name == '03' else 0
    ECGsignal = sig[:, signal_index].flatten()

    if F_name == '08':
        ECGsignal = -ECGsignal

    ECG_signal_filtered = nk.signal_filter(ECGsignal, sampling_rate=250, highcut=20, lowcut=1, method='butterworth')

    # Normalize the entire ECG_signal_downsampled array
    ECG_signal_filtered = (ECG_signal_filtered - ECG_signal_filtered.min()) / (ECG_signal_filtered.max() - ECG_signal_filtered.min())

    signal_annotation = wfdb.rdann(f'BIDMC/{F_name}', 'ecg')
    Label = np.array(signal_annotation.symbol)
    Sample = signal_annotation.sample

    for i, (label, sample) in enumerate(zip(Label, Sample)):
        if label in AAMI_BIDMC.keys():
            if i == 0 or i == len(Label) - 1:
                continue

            if i < len(Sample) - 1:
                next_r_peak = int(Sample[i + 1])
                if next_r_peak - sample < 80:
                    continue

            if 80 < sample < len(ECG_signal_filtered) - 80:
                ECG_segment = ECG_signal_filtered[sample-80:sample+80].tolist()
                with open(f'{AAMI_BIDMC[label]}_BIDMC.csv', 'a', newline='\n') as f:
                    writer = csv.writer(f)
                    writer.writerow(ECG_segment)

# Mapping AAMI categories to label numbers
AAMI_labels = {'N': 0, 'S': 1, 'V': 2, 'r': 3}

# Combine CSV files into one and add a label column
with open('combined_ecg_data.csv', 'w', newline='\n') as combined_file:
    writer = csv.writer(combined_file)
    for label in AAMI_BIDMC.values():
        if label != 'N':  # Skip the 'N' class
            with open(f'{label}_BIDMC.csv', 'r', newline='\n') as f:
                reader = csv.reader(f)
                for row in reader:
                    writer.writerow(row + [AAMI_labels[label]])

# Assuming `ECG_signal_filtered` is the preprocessed signal (filtered and normalized) for ST-T or BIDMC
sampling_rate = 250  # No need to down-sample

# Remove DC offset by centering the signal around zero
ECG_signal_centered = ECG_signal_filtered - np.mean(ECG_signal_filtered)

# Calculate the frequency spectrum for the centered signal
frequencies = fftfreq(len(ECG_signal_centered), 1 / sampling_rate)
spectrum = np.abs(fft(ECG_signal_centered))

# Plot the frequency spectrum
plt.plot(frequencies[:len(frequencies)//2], spectrum[:len(spectrum)//2])
plt.title('Frequency Spectrum of Preprocessed Signal (250 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()