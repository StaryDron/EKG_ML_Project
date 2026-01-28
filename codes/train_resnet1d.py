import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm


from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# ---------- ResNet1D ----------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        # dopasowanie skipa jeśli zmienia się liczba kanałów lub stride
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, n_classes=5, base_channels=64):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 4 "stages" jak w ResNet (ale 1D)
        self.layer1 = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels, stride=1),
            ResidualBlock1D(base_channels, base_channels, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels*2, stride=2),
            ResidualBlock1D(base_channels*2, base_channels*2, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(base_channels*2, base_channels*4, stride=2),
            ResidualBlock1D(base_channels*4, base_channels*4, stride=1),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock1D(base_channels*4, base_channels*8, stride=2),
            ResidualBlock1D(base_channels*8, base_channels*8, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        self.fc   = nn.Linear(base_channels*8, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.fc(x)                # (B, n_classes)
        return x


# ---------- Train / Eval loops ----------
def run_epoch(model, loader, criterion, optimizer=None, device="cpu", n_classes=5):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    all_probs = []
    all_true = []

    # Dodajemy tqdm do pętli po loaderze
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Batch", leave=False)

    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_n += y.size(0)

        # Aktualizacja paska postępu o aktualny loss
        current_loss = total_loss / total_n
        pbar.set_postfix(loss=f"{current_loss:.4f}")

        if not train:
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy())

    avg_loss = total_loss / total_n
    acc = total_correct / total_n

    # Na treningu nie musisz liczyć AUC/PR/F1 (wolne i mniej informacyjne)
    if train:
        return avg_loss, acc, None, None, None

    # ---- eval metrics ----
    y_true = np.concatenate(all_true, axis=0)           # (N,)
    y_prob = np.concatenate(all_probs, axis=0)          # (N, K)

    # F1 macro
    y_pred = y_prob.argmax(axis=1)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # One-hot dla AUC/PR-AUC
    y_true_oh = np.eye(n_classes)[y_true]               # (N, K)

    # ROC AUC (macro, OVR) + PR AUC (macro)
    # UWAGA: jeśli w walidacji/test brakuje jakiejś klasy, sklearn może rzucić wyjątek -> łapiemy
    try:
        roc_auc = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true_oh, y_prob, average="macro")
    except ValueError:
        pr_auc = float("nan")

    return avg_loss, acc, roc_auc, pr_auc, f1_macro


# ---------- Main example ----------
if __name__ == "__main__":
    # import Twoich loaderów
    import pandas as pd
    from data_ptbxl import make_dataloaders

    BASE_DIR = os.getcwd()
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    ART_DIR = os.path.join(PROJECT_ROOT_DIR, "artifacts")

    EXP_DIR = os.path.join(ART_DIR, "resnet_standard")
    os.makedirs(EXP_DIR, exist_ok=True)

    PTB_DIR = os.path.join(PROJECT_ROOT_DIR, "PTB-XL")

    train_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_train.csv"))
    val_df   = pd.read_csv(os.path.join(ART_DIR, "ptbxl_val.csv"))
    test_df  = pd.read_csv(os.path.join(ART_DIR, "ptbxl_test.csv"))

    train_loader, val_loader, test_loader, pipe, info = make_dataloaders(
        train_df, val_df, test_df,
        ptb_dir=PTB_DIR,
        batch_size=64,
        num_workers=0,
        use_derivative=True,
        use_augment=False
    )
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    model = ResNet1D(in_channels=info["in_channels"], n_classes=info["n_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_roc": []
    }

    for epoch in range(1, 12):
        train_loss, train_acc, _, _, _ = run_epoch(
                model, train_loader, criterion,
                optimizer=optimizer, device=device, n_classes=info["n_classes"]
            )

        val_loss, val_acc, val_roc, val_pr, val_f1 = run_epoch(
                model, val_loader, criterion,
                optimizer=None, device=device, n_classes=info["n_classes"]
            )

        # NOWOŚĆ: Zbieranie danych do historii
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_roc"].append(val_roc)

        print(
                f"Epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} "
                f"roc_auc {val_roc:.4f} pr_auc {val_pr:.4f} f1 {val_f1:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "resnet1d_best.pt"))

    import json

    with open(os.path.join(EXP_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    # test na najlepszym
    model.load_state_dict(torch.load(os.path.join(EXP_DIR, "resnet1d_best.pt"), map_location=device))
    test_loss, test_acc, test_roc, test_pr, test_f1 = run_epoch(
    model, test_loader, criterion, optimizer=None, device=device, n_classes=info["n_classes"])
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} roc_auc {test_roc:.4f} pr_auc {test_pr:.4f} f1 {test_f1:.4f}")

