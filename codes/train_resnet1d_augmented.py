import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
import json

from data_ptbxl import make_dataloaders
from train_resnet1d import ResNet1D, run_epoch

def main():
    BASE_DIR = os.getcwd()
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    ART_DIR = os.path.join(PROJECT_ROOT_DIR, "artifacts")

    EXP_DIR = os.path.join(ART_DIR, "resnet_augmented")
    os.makedirs(EXP_DIR, exist_ok=True)

    PTB_DIR = os.path.join(PROJECT_ROOT_DIR, "PTB-XL")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchomione z augmentacją na: {DEVICE}")

    train_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_train.csv"))
    val_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_val.csv"))
    test_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_test.csv"))

    train_loader, val_loader, test_loader, pipe, info = make_dataloaders(
        train_df, val_df, test_df,
        ptb_dir=PTB_DIR,
        batch_size=64,
        num_workers=0,
        use_derivative=True,  #dodanie pochodnej 
        use_augment=True  #augmentacja
    )

    model = ResNet1D(in_channels=info["in_channels"], n_classes=info["n_classes"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_roc": []
    }

    best_val_roc = 0.0
    epochs =15  
    patience = 4  
    counter = 0

    for epoch in range(1, epochs + 1):
        t_loss, t_acc, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, info["n_classes"]
        )
        v_loss, v_acc, v_roc, v_pr, v_f1 = run_epoch(
            model, val_loader, criterion, None, DEVICE, info["n_classes"]
        )

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_roc"].append(v_roc)

        print(f"Epoka {epoch:02d}/{epochs} | Val Acc: {v_acc:.4f} | Val ROC: {v_roc:.4f}")

        if v_roc > best_val_roc:
            best_val_roc = v_roc
            counter = 0 
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "resnet_aug_best.pt"))
            print(f" ---> Nowy najlepszy model zapisany (ROC: {v_roc:.4f})")
        else:
            counter += 1
            print(f" ---> Brak poprawy od {counter} epok.")

        #warunek przerwania
        if counter >= patience:
            print(f"\n[EARLY STOPPING] Przerwano trening w epoce {epoch}. Najlepszy ROC: {best_val_roc:.4f}")
            break

    with open(os.path.join(EXP_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    model.load_state_dict(torch.load(os.path.join(EXP_DIR, "resnet_aug_best.pt")))
    test_res = run_epoch(model, test_loader, criterion, None, DEVICE, info["n_classes"])
    print(f"\n--- WYNIK KOŃCOWY (TEST) ---")
    print(f"ROC-AUC: {test_res[2]:.4f} | F1: {test_res[4]:.4f}")


if __name__ == "__main__":
    main()
