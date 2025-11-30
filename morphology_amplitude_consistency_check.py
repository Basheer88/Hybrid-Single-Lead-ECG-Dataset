import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to compute R-peak heights and dynamically calculated QRS widths
def extract_morphological_features(ecg_signal, sampling_rate):
    # Ensure the signal length is sufficient
    if len(ecg_signal) < 100:
        print(f"Skipping short signal of length {len(ecg_signal)}")
        return [], []

    try:
        # Detect R-peaks using SciPy's find_peaks
        distance = int(0.2 * sampling_rate)  # 0.2 seconds minimum distance between peaks
        peaks, _ = find_peaks(ecg_signal, distance=distance, prominence=0.5)  # Adjust `prominence` as needed
        
        rpeak_heights = ecg_signal[peaks]  # R-peak heights
        qrs_widths = []
        
        # Calculate QRS width for each R-peak
        for i in peaks:
            # Find Q point (minimum point) before the R-peak
            q_start = max(i - int(0.04 * sampling_rate), 0)  # 40 ms before R-peak
            q_point = q_start + np.argmin(ecg_signal[q_start:i])  # Find minimum between q_start and R-peak

            # Find S point (minimum point) after the R-peak
            s_end = min(i + int(0.04 * sampling_rate), len(ecg_signal) - 1)  # 40 ms after R-peak
            s_point = i + np.argmin(ecg_signal[i:s_end])  # Find minimum between R-peak and s_end
            
            # Calculate QRS width as distance between Q and S points
            qrs_width = s_point - q_point
            qrs_widths.append(qrs_width)
        
        return rpeak_heights, qrs_widths

    except Exception as e:
        print(f"Error processing signal of length {len(ecg_signal)}: {e}")
        return [], []
    

# Load preprocessed data from CSV files for each dataset
datasets = {
    'MIT-BIH': {'file': 'MITBIH/combined_ecg_data.csv', 'sampling_rate': 250},
    'ST-T': {'file': 'ST-T/combined_ecg_data.csv', 'sampling_rate': 250},
    'BIDMC': {'file': 'BIDMC/combined_ecg_data.csv', 'sampling_rate': 250}
}

all_rpeak_heights = {}
all_qrs_widths = {}

for name, data in datasets.items():
    # Load the ECG signals, assuming each row contains 160 samples + 1 label
    ecg_data = pd.read_csv(data['file'], header=None)
    
    rpeak_heights_all = []
    qrs_widths_all = []
    
    # Loop through each row (each ECG signal) in the CSV file
    for _, row in ecg_data.iterrows():
        ecg_signal = np.array(row[:-1])  # Select the first 160 elements as the ECG signal, ignoring the label
        
        # Extract morphological features with error handling
        rpeak_heights, qrs_widths = extract_morphological_features(ecg_signal, data['sampling_rate'])
        
        # Append the extracted features to the lists
        rpeak_heights_all.extend(rpeak_heights)
        qrs_widths_all.extend(qrs_widths)
    
    # Store results for this dataset
    all_rpeak_heights[name] = rpeak_heights_all
    all_qrs_widths[name] = qrs_widths_all

# Increase transparency and bin count
alpha_value = 0.6
bin_count = 50  # Increased number of bins for granularity

# Solution 1: Separate Plots for Each Dataset
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, (name, rpeak_heights) in enumerate(all_rpeak_heights.items()):
    ax = axes[0, i]
    ax.hist(rpeak_heights, bins=bin_count, alpha=alpha_value, color='blue', edgecolor='black')
    ax.set_title(f'R-Peak Heights Distribution for {name}')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Count')
    
for i, (name, qrs_widths) in enumerate(all_qrs_widths.items()):
    ax = axes[1, i]
    ax.hist(qrs_widths, bins=bin_count, alpha=alpha_value, color='blue', edgecolor='black')
    ax.set_title(f'QRS Width Distribution for {name}')
    ax.set_xlabel('Width (samples)')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Solution 2 & 3: Combined Plot with Increased Granularity and Normalized Counts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot normalized R-Peak Heights distribution
for name, rpeak_heights in all_rpeak_heights.items():
    ax1.hist(rpeak_heights, bins=bin_count, alpha=alpha_value, label=name, density=True, edgecolor='black')
ax1.set_title('Normalized R-Peak Heights Distribution')
ax1.set_xlabel('Amplitude')
ax1.set_ylabel('Density')
ax1.legend()

# Plot normalized QRS Width distribution
for name, qrs_widths in all_qrs_widths.items():
    ax2.hist(qrs_widths, bins=bin_count, alpha=alpha_value, label=name, density=True, edgecolor='black')
ax2.set_title('Normalized QRS Width Distribution')
ax2.set_xlabel('Width (samples)')
ax2.set_ylabel('Density')
ax2.set_ylim(0, 30)  # Adjusted y-axis limit to reduce bidmc from 50 to 40 to make others more visual and overall better
ax2.legend()

plt.tight_layout()
plt.show()