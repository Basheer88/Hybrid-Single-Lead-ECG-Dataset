import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# Load the balanced and augmented dataset
BALANCED_DATASET_CSV = 'balanced_augmented_ecg_data.csv'
df = pd.read_csv(BALANCED_DATASET_CSV)

# Extract features and labels
X = df.iloc[:, :-1].values  # All columns except the last one are features
y = df['label'].values      # Last column is the label

# 1. Check Class Distribution
def plot_class_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title("Class Distribution After Resampling and Augmentation")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.show()

plot_class_distribution(y)

# 2. Check Feature Distribution - R-Peak Heights and QRS Widths
# Assume the signals are segmented with fixed-length samples (e.g., 160 samples per beat)
def extract_r_peak_heights_and_qrs_width(signal):
    # Identify R-peaks
    peaks, _ = find_peaks(signal, distance=30)  # Distance chosen based on sampling rate and expected heart rate
    if len(peaks) == 0:
        return None, None
    r_peak_heights = signal[peaks]

    # QRS width approximation (simplified for illustrative purposes)
    qrs_width = len(peaks)  # A rough estimate based on detected peaks

    return r_peak_heights, qrs_width

# Calculate R-peak heights and QRS widths for a subset of data
sample_size = 500  # Take a subset to speed up calculation
r_peak_heights_all = []
qrs_widths_all = []
for signal in X[:sample_size]:
    r_peaks, qrs_width = extract_r_peak_heights_and_qrs_width(signal)
    if r_peaks is not None:
        r_peak_heights_all.extend(r_peaks)
        qrs_widths_all.append(qrs_width)

# Plot the distributions of R-peak heights and QRS widths
def plot_morphological_features(r_peak_heights, qrs_widths):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(r_peak_heights, bins=30, kde=True)
    plt.title("Distribution of R-Peak Heights")
    plt.xlabel("Amplitude")

    plt.subplot(1, 2, 2)
    sns.histplot(qrs_widths, bins=20, kde=True)
    plt.title("Distribution of QRS Widths")
    plt.xlabel("Width (samples)")

    plt.tight_layout()
    plt.show()

plot_morphological_features(r_peak_heights_all, qrs_widths_all)

# 3. Data Augmentation Effectiveness - Visualize Augmented Signals
def plot_augmented_signals(original_signal, augmented_signal):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original_signal)
    plt.title("Original Signal")

    plt.subplot(2, 1, 2)
    plt.plot(augmented_signal)
    plt.title("Augmented Signal")

    plt.tight_layout()
    plt.show()

# Select a random sample to demonstrate augmentation
def augment_signal(signal):
    # Flip the signal with 50% probability
    if np.random.rand() > 0.5:
        signal = -signal

    # Add Gaussian noise
    noise = np.random.normal(0, 0.01, signal.shape)
    signal = signal + noise

    # Scale the signal by a random factor between 0.9 and 1.1
    scaling_factor = np.random.uniform(0.9, 1.1)
    signal = signal * scaling_factor

    return signal

random_idx = np.random.randint(0, X.shape[0])
original_signal = X[random_idx]
augmented_signal = augment_signal(original_signal)

plot_augmented_signals(original_signal, augmented_signal)

# Summary Statistics
print("Class Distribution After Resampling:")
print(df['label'].value_counts())
print("\nSummary Statistics of R-Peak Heights:")
print(pd.Series(r_peak_heights_all).describe())
print("\nSummary Statistics of QRS Widths:")
print(pd.Series(qrs_widths_all).describe())
