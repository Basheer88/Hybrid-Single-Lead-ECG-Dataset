import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score

torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CSV_FILE_PATH = 'balanced_augmented_ecg_data.csv'
df = pd.read_csv(CSV_FILE_PATH)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

class_counts = np.bincount(y)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Model definition
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    def forward(self, x):
        # x shape: (batch, channels, length)
        y = x.mean(dim=2)                 # global average pooling (batch, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.unsqueeze(2)                # (batch, channels, 1)
        return x * y                      # channel-wise scaling

class CNNClassifier(pl.LightningModule):
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
        self.attn = SEBlock(512, reduction=16)   # <--- Attention block after last conv
        self.fc1 = nn.Linear(512 * X.shape[1], 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.attn(x)                     # <--- apply attention here
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=weights)
        return loss
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels, weight=weights)
        pred = torch.argmax(outputs, dim=1)
        accuracy = Accuracy(task="multiclass", num_classes=4).to(self.device)
        precision = Precision(task="multiclass", num_classes=4).to(self.device)
        recall = Recall(task="multiclass", num_classes=4).to(self.device)
        f1 = F1Score(task="multiclass", num_classes=4).to(self.device)
        acc = accuracy(outputs, labels)
        prec = precision(pred, labels)
        rec = recall(pred, labels)
        f1_score_ = f1(pred, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=True)
        self.log('val_recall', rec, prog_bar=True)
        self.log('val_f1', f1_score_, prog_bar=True)
        return {"loss": loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1_score_}
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def evaluate_model(model_class, X_train, y_train, X_val, y_val):
    train_dataset = TensorDataset(torch.tensor(X_train).float().to(device), torch.tensor(y_train).long().to(device))
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(X_val).float().to(device), torch.tensor(y_val).long().to(device))
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0, shuffle=False)
    model = model_class().to(device)
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        callbacks=[ModelCheckpoint(monitor='val_loss'), LearningRateMonitor(logging_interval='epoch')],
        enable_progress_bar=True
    )
    trainer.fit(model, train_loader, val_loader)
    # Validation prediction
    y_preds, y_trues = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_preds.extend(pred)
            y_trues.extend(yb.cpu().numpy())
    acc = accuracy_score(y_trues, y_preds)
    prec = precision_score(y_trues, y_preds, average='weighted', zero_division=0)
    rec = recall_score(y_trues, y_preds, average='weighted', zero_division=0)
    f1 = f1_score(y_trues, y_preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1

results = []

# 1. Hold-Out
print("\n=== Hold-Out Validation ===")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
acc, prec, rec, f1 = evaluate_model(CNNClassifier, X_train, y_train, X_val, y_val)
results.append({'CV Scheme': 'Hold-Out', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1})

# 2. Stratified K-Fold
print("\n=== Stratified K-Fold ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, prec_list, rec_list, f1_list = [], [], [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}/5")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    acc, prec, rec, f1 = evaluate_model(CNNClassifier, X_train, y_train, X_val, y_val)
    acc_list.append(acc); prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)
results.append({'CV Scheme': 'Stratified K-Fold',
                'Accuracy': np.mean(acc_list), 'Precision': np.mean(prec_list), 'Recall': np.mean(rec_list), 'F1 Score': np.mean(f1_list)})

# 3. Standard K-Fold
print("\n=== Standard K-Fold ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list, prec_list, rec_list, f1_list = [], [], [], []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"  Fold {fold+1}/5")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    acc, prec, rec, f1 = evaluate_model(CNNClassifier, X_train, y_train, X_val, y_val)
    acc_list.append(acc); prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)
results.append({'CV Scheme': 'Standard K-Fold',
                'Accuracy': np.mean(acc_list), 'Precision': np.mean(prec_list), 'Recall': np.mean(rec_list), 'F1 Score': np.mean(f1_list)})

# Results Table
df_results = pd.DataFrame(results)
print("\nValidation Performance Comparison:")
print(df_results.to_string(index=False))
