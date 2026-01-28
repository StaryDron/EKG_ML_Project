import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Importy Twoich komponentów (upewnij się, że nazwy plików się zgadzają)
from train_resnet1d import ResNet1D
from data_ptbxl import make_dataloaders


def generate_interpretability_map(model, input_tensor, target_class):
    model.eval()
    input_tensor.requires_grad = True

    # Forward pass
    output = model(input_tensor)
    score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Pobranie gradientów i wygładzanie dla estetyki heatmapy
    gradients = input_tensor.grad.data.abs().cpu().numpy()[0]
    gradients = gaussian_filter1d(gradients, sigma=7, axis=1)  # sigma=7 daje ładniejsze plamy

    # Normalizacja [0, 1]
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
    return gradients


def plot_ecg_interpretability(signal, heatmap, title, save_path):
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, axes = plt.subplots(12, 2, figsize=(20, 25), sharex=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.15)
    time = np.arange(1000)

    for i in range(12):
        for col in [0, 1]:
            idx = i + (col * 12)
            ax = axes[i, col]
            sig = signal[idx]
            h = heatmap[idx]

            ax.plot(time, sig, color='black', lw=1.2, zorder=2)
            v_min, v_max = sig.min(), sig.max()
            # Nakładanie mapy ciepła w tle
            ax.imshow(h[np.newaxis, :], aspect='auto', cmap='YlOrRd',
                      alpha=0.4, extent=[0, 1000, v_min, v_max], zorder=1)

            name = leads[i] if col == 0 else f"deriv_{leads[i]}"
            ax.set_ylabel(name, rotation=0, labelpad=30, fontweight='bold', va='center')
            ax.set_yticks([])
            if i == 0:
                ax.set_title("Oryginalne Odprowadzenia" if col == 0 else "Pochodne (Derivative)", fontsize=14)

    plt.suptitle(title, fontsize=22, y=0.92)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.getcwd()
    ART_DIR = os.path.join(BASE_DIR, "..", "artifacts")
    PTB_DIR = os.path.join(BASE_DIR, "..", "PTB-XL")

    # Folder na wyniki
    OUT_DIR = os.path.join(ART_DIR, "interpretability_results")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Wczytanie modelu
    model = ResNet1D(in_channels=24, n_classes=5).to(device)
    model_path = os.path.join(ART_DIR, "resnet_augmented", "resnet_aug_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 2. Przygotowanie danych
    train_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_train.csv"))
    val_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_val.csv"))
    test_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_test.csv"))

    _, _, test_loader, _, _ = make_dataloaders(
        train_df, val_df, test_df, ptb_dir=PTB_DIR,
        batch_size=1, use_derivative=True, use_augment=False
    )

    # 3. Szukanie pacjentów dla każdej klasy
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    found_samples = {}  # label_idx -> (x_tensor, y_label)

    print("Przeszukiwanie zbioru testowego w celu znalezienia przykładów klas...")
    for x_batch, y_batch in test_loader:
        label = y_batch.item()
        if label not in found_samples:
            found_samples[label] = (x_batch, label)
            print(f"  > Znaleziono przykład dla klasy: {class_names[label]}")

        if len(found_samples) == 5:
            break

    # 4. Generowanie wykresów dla każdego znalezionego przypadku
    for label_idx, (x_tensor, _) in found_samples.items():
        name = class_names[label_idx]
        x_tensor = x_tensor.to(device)

        # Generowanie mapy
        heatmap = generate_interpretability_map(model, x_tensor, label_idx)
        signal_np = x_tensor.detach().cpu().numpy()[0]

        title = f"Analiza Interpretowalności - Klasa: {name}"
        save_path = os.path.join(OUT_DIR, f"interpretability_{name}.png")

        print(f"Generowanie wykresu dla {name}...")
        plot_ecg_interpretability(signal_np, heatmap, title, save_path)

    print(f"\nGotowe! Wszystkie 5 wykresów zapisano w: {OUT_DIR}")


if __name__ == "__main__":
    main()