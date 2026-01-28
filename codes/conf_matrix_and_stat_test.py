import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2

# Zakładamy, że klasy są w tych samych plikach co wcześniej
from train_resnet1d import ResNet1D
from data_ptbxl import make_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ART_DIR = "../artifacts"
    PTB_DIR = "../PTB-XL"
    OUT_DIR = os.path.join(ART_DIR, "final_results")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Wczytanie danych i modeli
    train_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_train.csv"))
    val_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_val.csv"))
    test_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_test.csv"))

    _, _, test_loader, _, _ = make_dataloaders(
        train_df, val_df, test_df, ptb_dir=PTB_DIR,
        batch_size=64, use_derivative=True, use_augment=False
    )

    # Model DL
    resnet = ResNet1D(in_channels=24, n_classes=5).to(device)
    resnet.load_state_dict(
        torch.load(os.path.join(ART_DIR, "resnet_augmented/resnet_aug_best.pt"), map_location=device))
    resnet.eval()

    # Model Baseline
    rf_model = joblib.load(os.path.join(ART_DIR, "baseline/baseline_rf_model.pkl"))
    X_test_rf = np.load(os.path.join(ART_DIR, "baseline/X_test_final.npy"))

    # 2. Zbieranie predykcji
    all_resnet_preds, all_rf_preds, all_true = [], [], []

    # DL preds
    with torch.no_grad():
        for x, y in test_loader:
            all_resnet_preds.append(resnet(x.to(device)).argmax(1).cpu().numpy())
            all_true.append(y.numpy())

    y_resnet = np.concatenate(all_resnet_preds)
    y_true = np.concatenate(all_true)
    y_rf = rf_model.predict(X_test_rf)

    # 3. Test McNemara (ResNet vs RF)
    # b: ResNet dobrze, RF źle | c: ResNet źle, RF dobrze
    b = np.sum((y_resnet == y_true) & (y_rf != y_true))
    c = np.sum((y_resnet != y_true) & (y_rf == y_true))

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = chi2.sf(stat, 1)

    print(f"--- TEST STATYSTYCZNY (McNemara) ---")
    print(f"P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("Różnica jest istotna statystycznie (p < 0.05)")

    # 4. Macierz Pomyłek dla najlepszego modelu
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    cm = confusion_matrix(y_true, y_resnet)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Znormalizowana Macierz Pomyłek - ResNet_Aug')
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_final.png"))
    print(f"Zapisano macierz pomyłek w {OUT_DIR}")


if __name__ == "__main__":
    main()