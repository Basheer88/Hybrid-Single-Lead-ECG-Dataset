import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision.models import resnet18


# ----- SETTINGS -----
CSV_FILE_PATH = 'balanced_augmented_ecg_data.csv'
BATCH_SIZE = 64
NUM_EPOCHS = 25
NUM_CLASSES = 4

# ----- DEVICE -----
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- DATA LOADING -----
df = pd.read_csv(CSV_FILE_PATH)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(int)

class_counts = np.bincount(y)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val_tensor = torch.tensor(X_val).float().to(device)
y_val_tensor = torch.tensor(y_val).long().to(device)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_length = X.shape[1]

# ----- MODEL ARCHITECTURES -----
class CNN5(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512 * input_length, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y, weight=weights)
        pred = torch.argmax(outputs, dim=1)
        acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(outputs, y)
        prec = Precision(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        rec = Recall(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class BiLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 5 BiLSTM layers, each with BatchNorm, ReLU, Dropout
        self.lstm1 = nn.LSTM(input_size=input_length, hidden_size=64, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(2*64)
        self.lstm2 = nn.LSTM(input_size=2*64, hidden_size=128, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(2*128)
        self.lstm3 = nn.LSTM(input_size=2*128, hidden_size=256, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(2*256)
        self.lstm4 = nn.LSTM(input_size=2*256, hidden_size=256, batch_first=True, bidirectional=True)
        self.bn4 = nn.BatchNorm1d(2*256)
        self.lstm5 = nn.LSTM(input_size=2*256, hidden_size=512, batch_first=True, bidirectional=True)
        self.bn5 = nn.BatchNorm1d(2*512)
        self.fc1 = nn.Linear(2*512, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm1(x)
        x = self.bn1(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.lstm2(x.unsqueeze(1))
        x = self.bn2(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.lstm3(x.unsqueeze(1))
        x = self.bn3(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.lstm4(x.unsqueeze(1))
        x = self.bn4(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.lstm5(x.unsqueeze(1))
        x = self.bn5(x[:, -1, :])
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y, weight=weights)
        pred = torch.argmax(outputs, dim=1)
        acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(outputs, y)
        prec = Precision(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        rec = Recall(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class BiGRU(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_length, hidden_size=64, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(2*64)
        self.gru2 = nn.GRU(input_size=2*64, hidden_size=128, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(2*128)
        self.gru3 = nn.GRU(input_size=2*128, hidden_size=256, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(2*256)
        self.gru4 = nn.GRU(input_size=2*256, hidden_size=256, batch_first=True, bidirectional=True)
        self.bn4 = nn.BatchNorm1d(2*256)
        self.gru5 = nn.GRU(input_size=2*256, hidden_size=512, batch_first=True, bidirectional=True)
        self.bn5 = nn.BatchNorm1d(2*512)
        self.fc1 = nn.Linear(2*512, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.gru1(x)
        x = self.bn1(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.gru2(x.unsqueeze(1))
        x = self.bn2(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.gru3(x.unsqueeze(1))
        x = self.bn3(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.gru4(x.unsqueeze(1))
        x = self.bn4(x[:, -1, :])
        x = F.relu(x)
        x, _ = self.gru5(x.unsqueeze(1))
        x = self.bn5(x[:, -1, :])
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y, weight=weights)
        pred = torch.argmax(outputs, dim=1)
        acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(outputs, y)
        prec = Precision(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        rec = Recall(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class TCN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512 * input_length, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y, weight=weights)
        pred = torch.argmax(outputs, dim=1)
        acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(outputs, y)
        prec = Precision(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        rec = Recall(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class ResNet1D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)
        # Adapt for 1D input (seq_len x 1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,1), stride=(2,1), padding=(3,0), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)  # [B, T] -> [B, 1, T, 1]
        x = self.resnet(x)
        x = self.dropout(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y, weight=weights)
        pred = torch.argmax(outputs, dim=1)
        acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(outputs, y)
        prec = Precision(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        rec = Recall(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(self.device)(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

# ----- TRAINING AND EVALUATION -----
def train_and_eval_all(models_dict):
    results = []
    for name, model_class in models_dict.items():
        print(f"\n=== Training {name} ===")
        model = model_class()
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(monitor='val_loss', save_last=True),
                LearningRateMonitor(logging_interval='epoch')
            ],
            default_root_dir=f'lightning_logs/{name}'
        )
        # Train
        train_dataset = TensorDataset(torch.tensor(X_train).float().to(device), torch.tensor(y_train).long().to(device))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
        trainer.fit(model, train_loader, val_loader)
        # Gather metrics
        metrics = trainer.callback_metrics
        results.append({
            'Model': name,
            'Accuracy': float(metrics['val_acc'].cpu().numpy()) * 100,
            'Precision': float(metrics['val_precision'].cpu().numpy()) * 100,
            'Recall': float(metrics['val_recall'].cpu().numpy()) * 100,
            'F1 Score': float(metrics['val_f1'].cpu().numpy()) * 100
        })
    return results

models_dict = {
    'CNN5': CNN5,
    'BiLSTM': BiLSTM,
    'BiGRU': BiGRU,
    'TCN': TCN,
    'ResNet1D': ResNet1D
}

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    results = train_and_eval_all(models_dict)
    df = pd.DataFrame(results).set_index('Model')
    print("\nValidation Performance Comparison (%):")
    print(df)
