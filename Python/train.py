#
# train.py
# Sample training script for HAR task.
#
# Copyright (c) 2025 MambaLite-Micro Authors
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba

class TinyMambaHAR(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=64, seq_len=10, num_classes=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear_in(x)      # [B, T, H]
        x = self.mamba(x)          # [B, T, H]
        x = x.transpose(1, 2)      # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

def load_har_data(data_dir):
    def load_txt(file_path):
        return pd.read_csv(file_path, delim_whitespace=True, header=None).values

    X_train = load_txt(os.path.join(data_dir, 'train', 'X_train.txt'))
    y_train = load_txt(os.path.join(data_dir, 'train', 'y_train.txt')).squeeze() - 1
    X_test = load_txt(os.path.join(data_dir, 'test', 'X_test.txt'))
    y_test = load_txt(os.path.join(data_dir, 'test', 'y_test.txt')).squeeze() - 1

    def prepare(X):
        X = F.pad(torch.tensor(X, dtype=torch.float32), (0, 570 - 561))
        return X.view(-1, 10, 57)

    X_train = prepare(X_train)
    X_test = prepare(X_test)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device='cuda'), yb.to(device='cuda')
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device='cuda'), yb.to(device='cuda')
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / len(dataloader), correct / total

def main():
    data_dir = r'path/to/UCI HAR Dataset'
    batch_size = 64
    epochs = 20
    lr = 1e-3
    hidden_dim = 64
    model_save_path = "path/to/output/mamba_har_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test = load_har_data(data_dir)
    val_len = int(0.2 * len(X_train))
    train_len = len(X_train) - val_len
    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [train_len, val_len])
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TinyMambaHAR(input_dim=57, hidden_dim=hidden_dim, seq_len=10, num_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved new best model to {model_save_path}")

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"[Final Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()
