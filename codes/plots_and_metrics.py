import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, accuracy_score, f1_score)

from train_resnet1d import ResNet1D
from inception_model import InceptionTime1D
from data_ptbxl import make_dataloaders


def get_nn_predictions(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_true.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)


def plot_curves(probs_dict, trues_dict, title, save_path, mode='ROC'):
    plt.figure(figsize=(8, 6))
    colors = {'train': '#1f77b4', 'val': '#2ca02c', 'test': '#d62728'}
    for split in ['train', 'val', 'test']:
        y_true = trues_dict[split]
        y_prob = probs_dict[split]
        n_classes = y_prob.shape[1]
        y_true_oh = np.eye(n_classes)[y_true]
        if mode == 'ROC':
            fpr, tpr, _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
            score = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[split], lw=2, label=f'{split} (AUC = {score:.3f})')
        else:
            precision, recall, _ = precision_recall_curve(y_true_oh.ravel(), y_prob.ravel())
            score = average_precision_score(y_true_oh, y_prob, average="macro")
            plt.plot(recall, precision, color=colors[split], lw=2, label=f'{split} (PR-AUC = {score:.3f})')
    plt.title(f'{mode} Curve: {title}')
    plt.legend(loc="lower left");
    plt.grid(alpha=0.3);
    plt.savefig(save_path, dpi=300);
    plt.close()


def main():
    BASE_DIR = os.getcwd()
    ART_DIR = os.path.join(BASE_DIR, "..", "artifacts")
    OUT_DIR = os.path.join(ART_DIR, "final_results")
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_train.csv"))
    val_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_val.csv"))
    test_df = pd.read_csv(os.path.join(ART_DIR, "ptbxl_test.csv"))
    train_l, val_l, test_l, _, _ = make_dataloaders(
        train_df, val_df, test_df, ptb_dir=os.path.join(BASE_DIR, "..", "PTB-XL"),
        batch_size=128, use_derivative=True, use_augment=False
    )
    loaders = {'train': train_l, 'val': val_l, 'test': test_l}

    model_configs = [
        {'name': 'ResNet_Std', 'class': ResNet1D, 'path': 'resnet_standard/resnet1d_best.pt', 'ch': 24},
        {'name': 'ResNet_Aug', 'class': ResNet1D, 'path': 'resnet_augmented/resnet_aug_best.pt', 'ch': 24},
        {'name': 'Inception_Std', 'class': InceptionTime1D, 'path': 'inception_standard/inception_standard_best.pt',
         'ch': 24},
        {'name': 'Inception_Aug', 'class': InceptionTime1D, 'path': 'inception_augmented/inception_augmented_best.pt',
         'ch': 24}
    ]

    metrics_list = []

    for cfg in model_configs:
        print(f"Ewaluacja: {cfg['name']}")
        model = cfg['class'](in_channels=cfg['ch'], n_classes=5).to(device)
        model.load_state_dict(torch.load(os.path.join(ART_DIR, cfg['path'])))
        probs, trues = {}, {}
        for split in ['train', 'val', 'test']:
            probs[split], trues[split] = get_nn_predictions(model, loaders[split], device)
            metrics_list.append({
                'Model': cfg['name'], 'Split': split,
                'Accuracy': accuracy_score(trues[split], np.argmax(probs[split], axis=1)),
                'F1_Macro': f1_score(trues[split], np.argmax(probs[split], axis=1), average='macro'),
                'ROC_AUC': auc(*roc_curve(np.eye(5)[trues[split]].ravel(), probs[split].ravel())[:2]),
                'PR_AUC': average_precision_score(np.eye(5)[trues[split]], probs[split], average="macro")
            })
        plot_curves(probs, trues, cfg['name'], os.path.join(OUT_DIR, f"{cfg['name']}_ROC.png"), 'ROC')
        plot_curves(probs, trues, cfg['name'], os.path.join(OUT_DIR, f"{cfg['name']}_PR.png"), 'PR')

        #wykresy uczenia z history.json
        hist_path = os.path.join(ART_DIR, os.path.dirname(cfg['path']), "history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                h = json.load(f)

            fig, ax1 = plt.subplots(figsize=(10, 6))

            color_loss = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss', color=color_loss)
            ax1.plot(range(1, len(h['train_loss']) + 1), h['train_loss'], color=color_loss, lw=2, label='Train Loss')
            ax1.tick_params(axis='y', labelcolor=color_loss)
            ax1.grid(alpha=0.3)

            if 'val_roc' in h:
                ax2 = ax1.twinx()
                color_roc = 'tab:orange'
                ax2.set_ylabel('Validation ROC-AUC', color=color_roc)
                ax2.plot(range(1, len(h['val_roc']) + 1), h['val_roc'], color=color_roc, lw=2, label='Val ROC-AUC')
                ax2.tick_params(axis='y', labelcolor=color_roc)
                ax2.set_ylim([min(h['val_roc']) - 0.05, 1.0])

            plt.title(f"Learning Progress: {cfg['name']}")
            fig.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{cfg['name']}_history.png"), dpi=300)
            plt.close()

    print("Ewaluacja: Baseline_RF")
    rf_model = joblib.load(os.path.join(ART_DIR, "baseline", "baseline_scaler.pkl")) 
    rf_model = joblib.load(os.path.join(ART_DIR, "baseline", "baseline_rf_model.pkl"))

    rf_probs, rf_trues = {}, {}
    for split in ['train', 'val', 'test']:
        X = np.load(os.path.join(ART_DIR, "baseline", f"X_{split}_final.npy"))
        y = np.load(os.path.join(ART_DIR, "baseline", f"y_{split}.npy"))
        p = rf_model.predict_proba(X)
        rf_probs[split], rf_trues[split] = p, y
        metrics_list.append({
            'Model': 'Baseline_RF', 'Split': split,
            'Accuracy': accuracy_score(y, np.argmax(p, axis=1)),
            'F1_Macro': f1_score(y, np.argmax(p, axis=1), average='macro'),
            'ROC_AUC': auc(*roc_curve(np.eye(5)[y].ravel(), p.ravel())[:2]),
            'PR_AUC': average_precision_score(np.eye(5)[y], p, average="macro")
        })
    plot_curves(rf_probs, rf_trues, 'Baseline_RF', os.path.join(OUT_DIR, "Baseline_RF_ROC.png"), 'ROC')
    plot_curves(rf_probs, rf_trues, 'Baseline_RF', os.path.join(OUT_DIR, "Baseline_RF_PR.png"), 'PR')

    pd.DataFrame(metrics_list).to_csv(os.path.join(OUT_DIR, "final_metrics_report.csv"), index=False)


if __name__ == "__main__":
    main()
