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

# Dataset
CSV_FILE_PATH = 'combined_ecg_data.csv'  # Update with your CSV file path

# Load the dataset
df = pd.read_csv(CSV_FILE_PATH)
# Adjusting the labels in the last column
df.iloc[:, -1] = df.iloc[:, -1]
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors and create datasets
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)  # Adjust the number of workers
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True)    # Adjust the number of workers

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * 160, 128)
        self.dropout = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(128, 2)  # two output class
        self.fc2 = nn.Linear(128, 4)  # four output class 

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
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
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)

        pred = torch.argmax(outputs, dim=1)  # Get predicted class labels

        # Create metric instances
        accuracy = Accuracy(task="multiclass", num_classes=4).to(self.device)
        precision = Precision(task="multiclass",num_classes=4).to(self.device)
        recall = Recall(task="multiclass",num_classes=4).to(self.device)
        f1 = F1Score(task="multiclass",num_classes=4).to(self.device)

        # Calculate and log metrics
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
        max_epochs=10,
        accelerator="auto", 
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
    # This block will only be executed if the script is run directly, not when it's imported
    train_and_evaluate(CNNClassifier)

