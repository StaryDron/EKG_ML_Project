import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import Pipeline

from custom_transformers import (
    ECGSignalLoader, ECGBandpassFilter,
    ECGAmplitudeClipper, ECGGlobalStandardizer,
    ECGAugmenter, ECGDerivativeAugmenter
)

LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]
label2id = {lab: i for i, lab in enumerate(LABELS)}

class ECGDataset(Dataset):
    def __init__(self, X_np, y_np, train=False, augment=None):
        """
        X_np: (N, C, T) numpy
        y_np: (N,) int numpy
        augment: callable, bierze (C,T) i zwraca (C,T) (np)
        """
        self.X = X_np.astype(np.float32)
        self.y = y_np.astype(np.int64)
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        y = self.y[idx]

        if self.train and self.augment is not None:
            # augment expects batch shape, więc owijamy w (1,C,T)
            x_aug = self.augment.transform(x[None, ...])[0]
            x = x_aug

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def build_signal_pipe(ptb_dir, use_derivative=False):
    steps = [
        ("load", ECGSignalLoader(ptb_dir=ptb_dir, filename_col="filename_lr")),
        ("bandpass", ECGBandpassFilter(lowcut=0.5, highcut=40.0, fs=100.0, order=4)),
        ("clip", ECGAmplitudeClipper(k=5.0))
    ]
    if use_derivative:
        steps.append(("deriv", ECGDerivativeAugmenter()))
    steps.append(("std", ECGGlobalStandardizer()))
    return Pipeline(steps)

def make_dataloaders(train_df, val_df, test_df, ptb_dir,
                     batch_size=64, num_workers=0,
                     use_derivative=False, use_augment=False):

    # --- y encoding ---
    y_train = train_df["label"].map(label2id).to_numpy()
    y_val   = val_df["label"].map(label2id).to_numpy()
    y_test  = test_df["label"].map(label2id).to_numpy()

    # --- preprocessing fit only on train ---
    pipe = build_signal_pipe(ptb_dir, use_derivative=use_derivative)
    X_train = pipe.fit_transform(train_df)  # (N,C,T)
    X_val   = pipe.transform(val_df)
    X_test  = pipe.transform(test_df)

    augment = None
    if use_augment:
        augment = ECGAugmenter(
            p_noise=0.5, noise_std=0.01,
            p_scale=0.5, scale_range=(0.9, 1.1),
            p_shift=0.5, max_shift=50,
            random_state=None
        )

    train_ds = ECGDataset(X_train, y_train, train=True,  augment=augment)
    val_ds   = ECGDataset(X_val,   y_val,   train=False, augment=None)
    test_ds  = ECGDataset(X_test,  y_test,  train=False, augment=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    info = {
        "label2id": label2id,
        "n_classes": len(LABELS),
        "in_channels": X_train.shape[1],
        "timesteps": X_train.shape[2]
    }
    return train_loader, val_loader, test_loader, pipe, info

