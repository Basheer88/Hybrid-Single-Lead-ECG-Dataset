# single layer cnn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Set 'Medium' will be faster but less precise, while 'high' will be more precise but potentially slower.
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
CSV_FILE_PATH = 'balanced_augmented_ecg_data.csv'  # Update with your balanced CSV file path
df = pd.read_csv(CSV_FILE_PATH)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Calculate class weights for weighted cross-entropy
class_counts = np.bincount(y)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert validation data to PyTorch tensors
X_val_tensor = torch.tensor(X_val).float().to(device)
y_val_tensor = torch.tensor(y_val).long().to(device)

# Define validation dataset and data loader
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        # Fully connected layers
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


# Define subset sizes
subset_sizes = [0.50]
n_splits = 10

# Dictionary to store metrics
metrics_dict = {
    'Subset Size': [],
    'Fold': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

def train_and_evaluate_with_kfold(model_class, subset_size, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_num = 1

    # Create subset
    if subset_size == 1.0:
        subset_X, subset_y = X, y
    else:
        subset_X, _, subset_y, _ = train_test_split(X, y, test_size=1 - subset_size, random_state=42, stratify=y)

    # K-fold Cross Validation
    for train_index, val_index in kf.split(subset_X, subset_y):
        X_train_fold, X_val_fold = subset_X[train_index], subset_X[val_index]
        y_train_fold, y_val_fold = subset_y[train_index], subset_y[val_index]

        # Convert fold data to PyTorch tensors
        train_dataset = TensorDataset(torch.tensor(X_train_fold).float().to(device), torch.tensor(y_train_fold).long().to(device))
        val_dataset = TensorDataset(torch.tensor(X_val_fold).float().to(device), torch.tensor(y_val_fold).long().to(device))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Initialize model
        model = model_class()

        # Train and evaluate
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="auto",
            devices=1,
            enable_model_summary=False,
            callbacks=[ModelCheckpoint(monitor='val_loss'), LearningRateMonitor(logging_interval='epoch')]
        )
        trainer.fit(model, train_loader, val_loader)

        # Retrieve metrics
        acc = trainer.callback_metrics['val_acc'].item()
        prec = trainer.callback_metrics['val_precision'].item()
        rec = trainer.callback_metrics['val_recall'].item()
        f1 = trainer.callback_metrics['val_f1'].item()

        # Store metrics for each fold
        metrics_dict['Subset Size'].append(subset_size)
        metrics_dict['Fold'].append(fold_num)
        metrics_dict['Accuracy'].append(acc)
        metrics_dict['Precision'].append(prec)
        metrics_dict['Recall'].append(rec)
        metrics_dict['F1 Score'].append(f1)

        print(f"Subset size {subset_size*100}%, Fold {fold_num} - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}")
        fold_num += 1

# Run training for each subset size
# Run K-fold cross-validation for each subset size
for size in subset_sizes:
    train_and_evaluate_with_kfold(CNNClassifier, size, n_splits)

# Plotting the metrics
# Calculate and plot the average performance for each subset
plt.figure(figsize=(12, 8))
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    avg_metric = []
    for size in subset_sizes:
        subset_metric = [metrics_dict[metric][i] for i, s in enumerate(metrics_dict['Subset Size']) if s == size]
        avg_metric.append(np.mean(subset_metric))
    plt.plot(subset_sizes, avg_metric, marker='o', label=f'Avg {metric}')

plt.xlabel('Subset Size')
plt.ylabel('Average Metric Value')
plt.title('Average Model Metrics across K-Folds for Different Dataset Subset Sizes')
plt.legend()
plt.grid(True)
plt.show()
