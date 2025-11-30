import wfdb
import numpy as np
import csv
import pywt
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Updated AAMI categorization
AAMI_MIT = {'N': 'Nfe/jnBLR', # Normal beats
            'S': 'SAJa',      # Supraventricular ectopic beats
            'V': 'VEr',       # Ventricular ectopic beats
            'F': 'F'}         # Fusion beats.

# Storage dictionary for 4 categories
AAMI = {'N': [], 'S': [], 'V': [], 'F': []}

file_name = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']

for F_name in file_name:
    record = wfdb.rdrecord(f'mitbihO/{F_name}', physical=True)
    # Check if MLII channel is present
    if 'MLII' not in record.sig_name:
        print(f"'MLII' channel not found in record {F_name}. Skipping...")
        continue
    mlII_index = record.sig_name.index('MLII')

    # Preprocess the entire record first
    ECG_signal_full = record.p_signal[:, mlII_index].flatten()
    # Filter signal by applying Butterworth bandpass filter using NeuroKit
    ECG_signal_filtered = nk.signal_filter(ECG_signal_full, sampling_rate=360, highcut=30, lowcut=1, method='butterworth', powerline=50)

    # ---------- Noise Reduction with DWT ----------
    # Decompose the filtered signal using DWT (choose suitable wavelet and level)
    coefficients  = pywt.wavedec(ECG_signal_filtered, wavelet='db8', level=8)

    # Apply thresholding for denoising (choose suitable thresholding method and parameters)
    # Estimate noise standard deviation (assuming data is noise)
    noise_std = np.median(np.abs(coefficients [-1])) / 0.6745

    # Estimate threshold using universal threshold formula
    threshold = noise_std * np.sqrt(2 * np.log(len(ECG_signal_filtered)))

    coefficients_den = [pywt.threshold(c, threshold, mode='soft') for c in coefficients]
    
    # Reconstruct the denoised signal
    # Reconstruct the signal
    reconstructed_signal = pywt.waverec(coefficients_den, 'db8')

    ECG_signal_downsampled = nk.signal_resample(reconstructed_signal, sampling_rate=360, desired_sampling_rate=250)

    # Normalize the entire ECG_signal_downsampled array
    ECG_signal_downsampled_normalized = (ECG_signal_downsampled - ECG_signal_downsampled.min()) / (ECG_signal_downsampled.max() - ECG_signal_downsampled.min())

    # Annotation
    signal_annotation = wfdb.rdann(f'mitbihO/{F_name}', "atr")
    Label = np.array(signal_annotation.symbol)
    Sample = signal_annotation.sample

    # Process each heartbeat
    for i, (label, sample) in enumerate(zip(Label, Sample)):
        if label in AAMI_MIT.keys():
            # Skip first and last samples
            if i == 0 or i == len(Label) - 1:
                continue

            # Adjust sample index due to downsampling
            r_peak = int(sample * (250 / 360))
            if 80 < r_peak < len(ECG_signal_downsampled_normalized) - 80:
                ECG_segment = ECG_signal_downsampled_normalized[r_peak-80:r_peak+80]
                AAMI[label].append(ECG_segment.tolist())

# Limit to the first 57,845 samples for 'N'
AAMI['N'] = AAMI['N'][:57845]

# Saving to CSV
for key, value in AAMI.items():
    with open(f'{key}_MITBIH.csv', 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(value)

# Mapping AAMI categories to label numbers
AAMI_labels = {'N': 0, 'S': 1, 'V': 2, 'F': 3}

# Combine CSV files into one and add a label column
with open('combined_ecg_data.csv', 'w', newline='\n') as combined_file:
    writer = csv.writer(combined_file)
    for key, value in AAMI.items():
        label = AAMI_labels[key]
        for row in value:
            writer.writerow(row + [label]) 

# Original signal frequency spectrum (before down-sampling)
frequencies_original = fftfreq(len(ECG_signal_filtered), 1 / 360)
spectrum_original = np.abs(fft(ECG_signal_filtered))

# After down-sampling, calculate frequency spectrum
frequencies_downsampled = fftfreq(len(ECG_signal_downsampled), 1 / 250)
spectrum_downsampled = np.abs(fft(ECG_signal_downsampled))

# Plotting frequency spectra before and after down-sampling
plt.figure(figsize=(12, 6))

# Original Signal Spectrum
plt.subplot(1, 2, 1)
plt.plot(frequencies_original[:len(frequencies_original)//2], spectrum_original[:len(spectrum_original)//2])
plt.title('Frequency Spectrum of Original Signal (360 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Down-sampled Signal Spectrum
plt.subplot(1, 2, 2)
plt.plot(frequencies_downsampled[:len(frequencies_downsampled)//2], spectrum_downsampled[:len(spectrum_downsampled)//2])
plt.title('Frequency Spectrum of Down-Sampled Signal (250 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()