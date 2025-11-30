import os
import pandas as pd
import numpy as np
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

# Load the balanced dataset created from the preprocessing script
CSV_FILE_PATH = 'balanced_augmented_ecg_data.csv'  # Update with your balanced CSV file path

# Load the dataset
df = pd.read_csv(CSV_FILE_PATH)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Calculate class weights for weighted cross-entropy
class_counts = np.bincount(y)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors and move data to the GPU
X_train_tensor = torch.tensor(X_train).float().to(device)
y_train_tensor = torch.tensor(y_train).long().to(device)
X_val_tensor = torch.tensor(X_val).float().to(device)
y_val_tensor = torch.tensor(y_val).long().to(device)

# Define datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, persistent_workers=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, persistent_workers=True, num_workers=2)

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * 160, 128)  # Adjust based on input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)  # Four output classes

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension if needed
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def train_and_evaluate(model_class):
    model = model_class()
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",  # Automatically uses GPU if available
        devices=1,
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),
            LearningRateMonitor(logging_interval='epoch')
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"{model_class.__name__} Validation Metrics:")
    print(f"- Accuracy: {trainer.callback_metrics['val_acc']}")
    print(f"- Precision: {trainer.callback_metrics['val_precision']}")
    print(f"- Recall: {trainer.callback_metrics['val_recall']}")
    print(f"- F1 Score: {trainer.callback_metrics['val_f1']}")

if __name__ == '__main__':
    train_and_evaluate(CNNClassifier)
