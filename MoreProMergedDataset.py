import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Path to the merged dataset CSV file
MERGED_DATASET_CSV = 'reduced_balanced_ecg_data.csv'
BALANCED_DATASET_CSV = 'balanced_ecg_data.csv'

# Load the merged dataset
df = pd.read_csv(MERGED_DATASET_CSV)
X = df.iloc[:, :-1].values  # Assuming all columns except the last one are features
y = df.iloc[:, -1].values   # Assuming the last column is the label

# Step 1: Resampling - Apply SMOTE and undersampling

# Apply SMOTE to oversample minority classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Optional: Apply undersampling on the majority class
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_resampled, y_resampled = undersample.fit_resample(X_resampled, y_resampled)

# Convert resampled data back to DataFrame for easier manipulation
df_resampled = pd.DataFrame(X_resampled)
df_resampled['label'] = y_resampled

# Step 2: Data Augmentation Functions
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

# Apply data augmentation to minority classes
df_augmented = df_resampled.copy()
for class_label in np.unique(y_resampled):
    # Identify the minority class samples for augmentation
    class_df = df_resampled[df_resampled['label'] == class_label]
    if len(class_df) < len(df_resampled[df_resampled['label'] == y_resampled.max()]):
        augmented_samples = class_df.apply(lambda row: augment_signal(row[:-1].values), axis=1)
        augmented_df = pd.DataFrame(augmented_samples.tolist(), columns=df_resampled.columns[:-1])
        augmented_df['label'] = class_label
        df_augmented = pd.concat([df_augmented, augmented_df], ignore_index=True)

# Shuffle the augmented dataset
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the final balanced and augmented dataset
df_augmented.to_csv(BALANCED_DATASET_CSV, index=False)

print(f"Balanced and augmented dataset saved to {BALANCED_DATASET_CSV}")
