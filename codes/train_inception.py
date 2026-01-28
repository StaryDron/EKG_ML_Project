import os
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import json
from data_ptbxl import make_dataloaders
from train_resnet1d import run_epoch
from inception_model import InceptionTime1D


def train_inception(use_aug):
    device = torch.device("cuda")
    suffix = "augmented" if use_aug else "standard"
    print(f"--- START: Inception1D ({suffix}) ---")

    BASE_DIR = os.getcwd()
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    ART_DIR = os.path.join(PROJECT_ROOT_DIR, "artifacts")
    EXP_DIR = os.path.join(ART_DIR, f"inception_{suffix}")
    os.makedirs(EXP_DIR, exist_ok=True)
    PTB_DIR = os.path.join(PROJECT_ROOT_DIR, "PTB-XL")

    train_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_train.csv"))
    val_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_val.csv"))
    test_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_test.csv"))

    train_loader, val_loader, test_loader, _, info = make_dataloaders(
        train_df, val_df, test_df, ptb_dir=PTB_DIR,
        batch_size=128, num_workers=4,
        use_derivative=True, use_augment=use_aug  # To jest kluczowa różnica
    )

    model = InceptionTime1D(in_channels=info["in_channels"], n_classes=info["n_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_roc": []}
    best_val_roc, patience, counter = 0.0, 5, 0

    for epoch in range(1, 21):
        t_loss, t_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, info["n_classes"])
        v_loss, v_acc, v_roc, v_pr, v_f1 = run_epoch(model, val_loader, criterion, None, device, info["n_classes"])

        history["train_loss"].append(t_loss)
        history["val_roc"].append(v_roc)
        print(f"E{epoch:02d} | Val ROC: {v_roc:.4f}")

        if v_roc > best_val_roc:
            best_val_roc = v_roc
            torch.save(model.state_dict(), os.path.join(EXP_DIR, f"inception_{suffix}_best.pt"))
            counter = 0
        else:
            counter += 1
        if counter >= patience: break

    with open(os.path.join(EXP_DIR, "history.json"), "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    train_inception(use_aug=False)  # Zmień na True w drugim pliku