# without Kfold
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # First convolutional layer with dilation
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)  # Standard conv
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SEBlock(32)
        
        # Second convolutional layer with higher dilation
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2)  # Moderate dilation
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SEBlock(64)

        # Standard convolution for deeper layers
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4)  # Increased dilation
        self.bn3 = nn.BatchNorm1d(128)
        self.se3 = SEBlock(128)

        # Additional convolutional layers without dilation
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)  # Standard conv
        self.bn4 = nn.BatchNorm1d(256)
        self.se4 = SEBlock(256)

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)  # Standard conv
        self.bn5 = nn.BatchNorm1d(512)
        self.se5 = SEBlock(512)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 160, 128)  # Adjust based on input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)  # Four output classes

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension if needed
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.se5(x)
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
#subset_sizes = [0.25, 0.50, 0.75, 1.0]
subset_sizes = [1.0]

# Dictionary to store metrics
metrics_dict = {
    'Subset Size': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

def train_and_evaluate_with_kfold(model_class, subset_size):
        # Create subset
        if subset_size == 1.0:
            subset_X, subset_y = X, y
        else:
            subset_X, _, subset_y, _ = train_test_split(X, y, test_size=1 - subset_size, random_state=42, stratify=y)

        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(torch.tensor(subset_X).float().to(device), torch.tensor(subset_y).long().to(device))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Initialize model
        model = model_class()

        # Train and evaluate
        trainer = pl.Trainer(
            max_epochs=100,
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
        metrics_dict['Accuracy'].append(acc)
        metrics_dict['Precision'].append(prec)
        metrics_dict['Recall'].append(rec)
        metrics_dict['F1 Score'].append(f1)

        print(f"Subset size {subset_size*100}%, - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}")

# Run training for each subset size
# Run K-fold cross-validation for each subset size
for size in subset_sizes:
    train_and_evaluate_with_kfold(CNNClassifier, size)

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
plt.title('Average Model Metrics for Different Dataset Subset Sizes')
plt.legend()
plt.grid(True)
plt.show()
