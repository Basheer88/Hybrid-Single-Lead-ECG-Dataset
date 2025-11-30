# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score
from scipy import stats

# Set 'Medium' will be faster but less precise, while 'high' will be more precise but potentially slower.
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Define the CNN model
class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * 160, 128)  # Adjust based on input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)  # Four output classes

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension if needed
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)  # Apply class weights here
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels, weight=weights)  # Apply class weights

        pred = torch.argmax(outputs, dim=1)
        # Calculate metrics
        accuracy = Accuracy(task="multiclass", num_classes=4).to(self.device)
        precision = Precision(task="multiclass", num_classes=4).to(self.device)
        recall = Recall(task="multiclass", num_classes=4).to(self.device)
        f1 = F1Score(task="multiclass", num_classes=4).to(self.device)

        acc = accuracy(outputs, labels)
        prec = precision(pred, labels)
        rec = recall(pred, labels)
        f1_score = f1(pred, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1_score, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# List of dataset file paths, including the hybrid dataset
DATASET_PATHS = ['MIT.csv', 'BID.csv', 'STT.csv', 'balanced_augmented_ecg_data.csv']
n_splits = 10  # Number of folds for K-fold cross-validation

# Dictionary to store average metrics for each dataset
# Store metrics across datasets
metrics_summary = {
    'Dataset': [],
    'Mean Accuracy': [],
    'Std Accuracy': [],
    'Mean Precision': [],
    'Std Precision': [],
    'Mean Recall': [],
    'Std Recall': [],
    'Mean F1 Score': [],
    'Std F1 Score': []
}

# Loop through each dataset
all_metrics = {}  # Store metrics for each dataset for statistical comparison
for dataset_path in DATASET_PATHS:
    # Load the dataset
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Calculate class weights for weighted cross-entropy
    class_counts = np.bincount(y)
    class_weights = 1. / class_counts
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Set up K-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Metrics storage for each fold
    fold_metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    # K-fold Cross Validation
    for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Convert fold data to PyTorch tensors
        train_dataset = TensorDataset(torch.tensor(X_train_fold).float().to(device), torch.tensor(y_train_fold).long().to(device))
        val_dataset = TensorDataset(torch.tensor(X_val_fold).float().to(device), torch.tensor(y_val_fold).long().to(device))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Initialize and train model
        model = CNNClassifier()
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            enable_model_summary=False,
            callbacks=[ModelCheckpoint(monitor='val_loss'), LearningRateMonitor(logging_interval='epoch')]
        )
        trainer.fit(model, train_loader, val_loader)

        # Retrieve metrics for fold and append them
        acc = trainer.callback_metrics['val_acc'].item()
        prec = trainer.callback_metrics['val_precision'].item()
        rec = trainer.callback_metrics['val_recall'].item()
        f1 = trainer.callback_metrics['val_f1'].item()
        fold_metrics['Accuracy'].append(acc)
        fold_metrics['Precision'].append(prec)
        fold_metrics['Recall'].append(rec)
        fold_metrics['F1 Score'].append(f1)

        # Print metrics for each fold to examine consistency
        print(f"Fold {fold} - {dataset_path} Metrics:")
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    # Calculate mean metrics for this dataset
    # Calculate mean and standard deviation across folds
    metrics_summary['Dataset'].append(dataset_path)
    metrics_summary['Mean Accuracy'].append(np.mean(fold_metrics['Accuracy']))
    metrics_summary['Std Accuracy'].append(np.std(fold_metrics['Accuracy']))
    metrics_summary['Mean Precision'].append(np.mean(fold_metrics['Precision']))
    metrics_summary['Std Precision'].append(np.std(fold_metrics['Precision']))
    metrics_summary['Mean Recall'].append(np.mean(fold_metrics['Recall']))
    metrics_summary['Std Recall'].append(np.std(fold_metrics['Recall']))
    metrics_summary['Mean F1 Score'].append(np.mean(fold_metrics['F1 Score']))
    metrics_summary['Std F1 Score'].append(np.std(fold_metrics['F1 Score']))

    # Store fold metrics for statistical comparison
    all_metrics[dataset_path] = fold_metrics

    # Summary of dataset performance with standard deviation
    print(f"Summary for {dataset_path}:")
    print(f"  Mean Accuracy: {metrics_summary['Mean Accuracy'][-1]:.4f} ± {metrics_summary['Std Accuracy'][-1]:.4f}")
    print(f"  Mean Precision: {metrics_summary['Mean Precision'][-1]:.4f} ± {metrics_summary['Std Precision'][-1]:.4f}")
    print(f"  Mean Recall: {metrics_summary['Mean Recall'][-1]:.4f} ± {metrics_summary['Std Recall'][-1]:.4f}")
    print(f"  Mean F1 Score: {metrics_summary['Mean F1 Score'][-1]:.4f} ± {metrics_summary['Std F1 Score'][-1]:.4f}\n")

# Perform statistical tests between hybrid dataset and individual datasets
# Statistical tests between hybrid and individual datasets
print("\nStatistical Validation (Hybrid vs. Individual Datasets):")
hybrid_metrics = all_metrics['balanced_augmented_ecg_data.csv']
for dataset_path in DATASET_PATHS[:-1]:  # Exclude hybrid for comparison
    individual_metrics = all_metrics[dataset_path]
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        hybrid_values = hybrid_metrics[metric]
        individual_values = individual_metrics[metric]

        # Paired t-test or Wilcoxon test
        t_stat, p_value = stats.ttest_rel(hybrid_values, individual_values)
        if p_value >= 0.05:
            _, p_value = stats.wilcoxon(hybrid_values, individual_values)

        # Display comparison results
        print(f"Comparison for {metric} between Hybrid and {dataset_path}: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  => Statistically significant improvement for {metric} with the hybrid dataset.")
        else:
            print(f"  => No statistically significant difference for {metric}.")