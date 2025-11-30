import wfdb
import numpy as np
import csv
from scipy import signal
from scipy.signal import find_peaks
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def detect_inverted_r_waves(ecg_signal, fs):
    # Find peaks (possible R-waves)
    peaks, _ = find_peaks(ecg_signal, distance=0.2*fs)  # Assuming R-waves are at least 200 ms apart

    # Check if any R-wave is inverted
    inverted_r_waves = [peak for peak in peaks if ecg_signal[peak] < 0]
    return inverted_r_waves

# Adapt this dictionary based on the European ST-T annotations
# Example categories (these should be updated based on actual database annotations)
AAMI_EuroSTT = {
    'N': '•',  # Normal beat
    'S': 'S',  # Supraventricular premature or ectopic beat
    'V': 'V',  # Premature ventricular contraction
    'F': 'F'  # Fusion of ventricular and normal beat
}

AAMI = {'N': [], 'S': [], 'V': [], 'F': []}

# Example file names from European ST-T Database (these should be updated with actual file names)
file_name = ['e0103', 'e0104', 'e0105', 'e0106', 'e0107', 'e0108', 'e0110', 'e0111', 'e0112',
             'e0113', 'e0114', 'e0115', 'e0116', 'e0118', 'e0119', 'e0121', 'e0122', 'e0123',
             'e0124', 'e0125', 'e0126', 'e0127', 'e0129', 'e0133', 'e0136', 'e0139', 'e0147',
             'e0148', 'e0151', 'e0154', 'e0155', 'e0159', 'e0161', 'e0162', 'e0163', 'e0166',
             'e0170', 'e0202', 'e0203', 'e0204', 'e0205', 'e0206', 'e0207', 'e0208', 'e0210',
             'e0211', 'e0212', 'e0213', 'e0302', 'e0303', 'e0304', 'e0305', 'e0306', 'e0403',
             'e0404', 'e0405', 'e0406', 'e0408', 'e0409', 'e0410', 'e0411', 'e0413', 'e0415',
             'e0417', 'e0418', 'e0501', 'e0509', 'e0515', 'e0601', 'e0602', 'e0603', 'e0604',
             'e0605', 'e0606', 'e0607', 'e0609', 'e0610', 'e0611', 'e0612', 'e0613', 'e0614',
             'e0615', 'e0704', 'e0801', 'e0808', 'e0817', 'e0818', 'e1301', 'e1302', 'e1304'
             ]

for F_name in file_name:
    record = wfdb.rdrecord(f'st-t/{F_name}', physical=True)
    # Check if MLIII channel is present
    if 'MLIII' not in record.sig_name:
        print(f"'MLII' channel not found in record {F_name}. Skipping...")
        continue
    mlIII_index = record.sig_name.index('MLIII')

    # Preprocess the entire record first
    ECG_signal_full = record.p_signal[:, mlIII_index].flatten()
    # Filter signal by applying Butterworth bandpass filter using NeuroKit
    ECG_signal_filtered = nk.signal_filter(ECG_signal_full, sampling_rate=250, highcut=20, lowcut=1, method='butterworth')
    ECG_signal_filtered = (ECG_signal_filtered - ECG_signal_filtered.min()) / (ECG_signal_filtered.max() - ECG_signal_filtered.min())

    # Check if any of the detected R-waves are inverted
    inverted_r_waves = detect_inverted_r_waves(ECG_signal_filtered, 250)
    if inverted_r_waves:
        print(f"Inverted R-waves detected in file hereeeeeeee {F_name}")
        ECG_signal_filtered *= -1

    # Annotation
    signal_annotation = wfdb.rdann(f'st-t/{F_name}', "atr")
    Label = np.array(signal_annotation.symbol)
    Sample = signal_annotation.sample

    # Process each heartbeat
    for i, (label, sample) in enumerate(zip(Label, Sample)):
        if label in AAMI_EuroSTT.keys():
            # Skip first and last samples
            if i == 0 or i == len(Label) - 1:
                continue

            r_peak = int(sample)
            # Ensure no other R-peak within the segment
            if i < len(Sample) - 1:  # Check if there is a next sample
                next_r_peak = int(Sample[i + 1])
                if next_r_peak - r_peak < 90:
                    continue  # Skip if next R-peak is too close

            if 80 < r_peak < len(ECG_signal_filtered) - 80:
                ECG_segment = ECG_signal_filtered[r_peak-80:r_peak+80]
                AAMI[label].append(ECG_segment.tolist())

# Saving to CSV
for key, value in AAMI.items():
    with open(f'{key}_euro_stt.csv', 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(value)

# Mapping AAMI categories to label numbers
AAMI_labels = {'N': 0, 'S': 1, 'V': 2, 'F': 3}

# Combine CSV files into one and add a label column
with open('combined_ecg_data.csv', 'w', newline='\n') as combined_file:
    writer = csv.writer(combined_file)
    for key, value in AAMI.items():
        if key != 'N':  # Skip the 'N' class
            label = AAMI_labels[key]
            for row in value:
                writer.writerow(row + [label])


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