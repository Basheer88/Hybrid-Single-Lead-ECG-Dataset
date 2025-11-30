import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


# Set 'Medium' will be faster but less precise, while 'high' will be more precise but potentially slower.
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------------------------------------------------------
# 1) Model definitions
# -----------------------------------------------------------------------------
class CNN5(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: [B, T]
        return self.net(x.unsqueeze(1))

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log('val_loss', nn.CrossEntropyLoss()(logits, y), prog_bar=True)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }


class LSTMClassifier(pl.LightningModule):
    def __init__(self, n_classes, hidden_dim=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(-1)  # [B, T, 1]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        # log it so Trainer shows it in the progress bar
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class GRUClassifier(LSTMClassifier):
    def __init__(self, n_classes, hidden_dim=128, num_layers=3):
        super().__init__(n_classes, hidden_dim, num_layers)
        # overwrite LSTM with GRU
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


from torchvision.models import resnet18
class ResNet1D(pl.LightningModule):
    def __init__(self, n_classes, seq_len):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        # adapt conv1 to accept [B,1,seq_len,1]
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)
        self.seq_len = seq_len

    def forward(self, x):
        # x: [B, T] → [B, 1, T, 1]
        x = x.unsqueeze(1).unsqueeze(-1)
        return self.backbone(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Remove extra padding at the end
        return self.relu(self.conv(x))[:, :, : -self.conv.padding[0]]


class TCNClassifier(pl.LightningModule):
    def __init__(self, n_classes, seq_len, channels=[16, 32, 64], kernel_size=5):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers.append(TCNBlock(in_ch, ch, kernel_size, dilation=1))
            in_ch = ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1] * seq_len, n_classes)

    def forward(self, x):
        # x: [B, T] → [B, 1, T]
        x = x.unsqueeze(1)
        out = self.network(x)
        return self.fc(out.flatten(1))

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# -----------------------------------------------------------------------------
# 2) Main training & evaluation script
# -----------------------------------------------------------------------------
def main():
    # Load & preprocess data
    df = pd.read_csv('balanced_augmented_ecg_data.csv')
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)

    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    seq_len = X.shape[1]

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_val_bin = label_binarize(y_val, classes=classes)

    # DataLoaders (num_workers=0 on Windows)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train), 
            torch.from_numpy(y_train).long()
        ),
        batch_size=64, shuffle=True, num_workers=3
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val), 
            torch.from_numpy(y_val).long()
        ),
        batch_size=64, shuffle=False, num_workers=3
    )

    # Define models to train
    model_classes = {
        #'CNN5':   CNN5,
        'BiLSTM': LSTMClassifier,
        'BiGRU':  GRUClassifier,
        #'ResNet': ResNet1D,
        'TCN':    TCNClassifier
    }

    y_scores = {}
    y_preds = {}

    # Train & collect predictions
    for name, Cls in model_classes.items():
        print(f'\n=== Training {name} ===')
        model = Cls(n_classes=n_classes, seq_len=seq_len) \
            if name in ('ResNet', 'TCN') \
            else Cls(n_classes=n_classes)


        # Create a NEW Trainer for each model
        trainer = pl.Trainer(
            max_epochs=25,
            accelerator='auto',
            devices=1,
            callbacks=[LearningRateMonitor('epoch')],
            enable_progress_bar=True,
            default_root_dir=f'lightning_logs/{name}'   # unique folder per model
        )

        trainer.fit(model, train_loader, val_loader)

        # Validation: get probabilities and predicted labels
        probs_list = []
        preds_list = []
        for xb, _ in val_loader:
            xb = xb.to(model.device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            probs_list.append(probs)
            preds_list.append(preds)

        y_scores[name] = np.vstack(probs_list)
        y_preds[name]  = np.concatenate(preds_list)

    # Compute & print Accuracy, Precision, Recall, F1
    results = []
    for name, preds in y_preds.items():
        acc  = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average='weighted', zero_division=0)
        rec  = recall_score(y_val, preds, average='weighted', zero_division=0)
        f1   = f1_score(y_val, preds, average='weighted', zero_division=0)
        results.append({
            'Model':    name,
            'Accuracy': f'{acc*100:.2f}',
            'Precision':f'{prec*100:.2f}',
            'Recall':   f'{rec*100:.2f}',
            'F1 Score': f'{f1*100:.2f}'
        })

    df_res = pd.DataFrame(results).set_index('Model')
    print('\nValidation Performance Comparison:')
    print(df_res)


if __name__ == '__main__':
    # On Windows, for multiprocessing safety
    from multiprocessing import freeze_support
    freeze_support()
    main()
