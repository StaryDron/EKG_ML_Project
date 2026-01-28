import os
import numpy as np
import pandas as pd
import wfdb

from sklearn.base import BaseEstimator, TransformerMixin


BASE_DIR = os.getcwd()

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

PTB_DIR = os.path.join(PROJECT_ROOT_DIR, "PTB-XL")

DATABASE_PATH= os.path.join(PTB_DIR,"ptbxl_database.csv")

STATEMENTS_PATH= os.path.join(PTB_DIR,"scp_statements.csv")



print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT_DIR:", PROJECT_ROOT_DIR)
print("PTB_DIR:", PTB_DIR)
print("DATABASE_PATH:", DATABASE_PATH)
print("STATEMENTS_PATH:",STATEMENTS_PATH)


class LabelBuilder(BaseEstimator, TransformerMixin):
    """
    Input:  DataFrame z kolumną 'scp_codes' + DataFrame st (scp_statements)
    Output: ten sam DataFrame, ale z dodaną kolumną 'label'
    """
    def __init__(self, st_df, allowed_labels=None):
        self.st_df = st_df.set_index("Unnamed: 0")
        if allowed_labels is None:
            allowed_labels = ["NORM", "MI", "STTC", "HYP", "CD"]
        self.allowed_labels = allowed_labels

    def _scp_to_classes(self, scp_codes_str):
        if pd.isna(scp_codes_str):
            return []
        scp_dict = eval(scp_codes_str)  # w PTB-XL tak jest zapisane
        classes = set()
        for scp in scp_dict.keys():
            if scp in self.st_df.index and self.st_df.loc[scp, "diagnostic"] == 1:
                classes.add(self.st_df.loc[scp, "diagnostic_class"])
        return list(classes)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["diag_classes"] = X["scp_codes"].apply(self._scp_to_classes)
        X["label"] = X["diag_classes"].apply(
            lambda lst: lst[0] if len(lst) > 0 else np.nan
        )
        X = X[X["label"].isin(self.allowed_labels)].copy()
        return X
    

class ECGSignalLoader(BaseEstimator, TransformerMixin):
    """
    Input:  DataFrame z kolumną filename_lr
    Output: numpy array o kształcie (n_records, n_leads, n_samples)
            czyli tensor gotowy np. pod modele DL.

    Opcjonalnie można wybrać podzbiór odprowadzeń po nazwie (lead_names).
    """

    def __init__(self, ptb_dir, filename_col="filename_lr", lead_names=None):
        self.ptb_dir = ptb_dir
        self.filename_col = filename_col
        self.lead_names = lead_names  # np. ["I", "II", "V1"] albo None = wszystkie

    def fit(self, X, y=None):
        """
        Jeżeli chcemy wybierać konkretne odprowadzenia po nazwie,
        to w fit zapamiętamy indeksy tych odprowadzeń na podstawie pierwszego rekordu.
        """
        if self.lead_names is not None:
            first_path = os.path.join(self.ptb_dir, X[self.filename_col].iloc[0])
            sig, info = wfdb.rdsamp(first_path)
            name_to_idx = {name: i for i, name in enumerate(info["sig_name"])}
            self.lead_indices_ = [name_to_idx[name] for name in self.lead_names]
        else:
            self.lead_indices_ = None
        return self

    def transform(self, X, y=None):
        signals = []

        for fname in X[self.filename_col]:
            path = os.path.join(self.ptb_dir, fname)
            sig, info = wfdb.rdsamp(path)   # sig: (n_samples, n_leads)

            # wybór konkretnych odprowadzeń (jeśli ustawione)
            if self.lead_indices_ is not None:
                sig = sig[:, self.lead_indices_]

            # chcemy (n_leads, n_samples), więc transpozycja:
            sig = sig.T

            signals.append(sig)

        # kształt: (n_records, n_leads, n_samples)
        X_out = np.stack(signals, axis=0)
        return X_out
    
    
from scipy.signal import butter, filtfilt

class ECGBandpassFilter(BaseEstimator, TransformerMixin):
    """
    Filtruje sygnał EKG filtrem Butterwortha w paśmie [lowcut, highcut].

    Input:  X o kształcie (n_records, n_leads, n_timesteps)
    Output: X o tym samym kształcie, ale przefiltrowane wzdłuż osi czasu.
    """

    def __init__(self, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def fit(self, X, y=None):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.b_, self.a_ = butter(self.order, [low, high], btype="band")
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        # filtrujemy wzdłuż osi czasu (ostatnia oś)
        X_filt = filtfilt(self.b_, self.a_, X, axis=-1)
        return X_filt


class ECGAmplitudeClipper(BaseEstimator, TransformerMixin):
    """
    Przycina amplitudy do przedziału [-k * std_global, k * std_global]
    liczonych na trainie (per lead).
    Input:  X: (N, L, T)
    """

    def __init__(self, k=5.0):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X)
        # globalne std per lead
        self.std_ = X.std(axis=(0, 2))
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        std = self.std_[None, :, None]
        lower = -self.k * std
        upper = self.k * std
        return np.clip(X, lower, upper)

class ECGGlobalStandardizer(BaseEstimator, TransformerMixin):
    """
    Globalna normalizacja:
    - liczy mean/std dla każdego leada na zbiorze treningowym
      (po wszystkich rekordach i po czasie),
    - stosuje te same parametry do train/val/test.

    Input:  X: (n_records, n_leads, n_timesteps)
    Output: X o tym samym kształcie.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X)
        # średnia i std per lead po wszystkich rekordach i po czasie
        # axis=(0,2) -> agregujemy po rekordach i po czasie, zostaje (n_leads,)
        self.mean_ = X.mean(axis=(0, 2))
        self.std_ = X.std(axis=(0, 2)) + self.eps
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        mean = self.mean_[None, :, None]  # broadcast: (1, n_leads, 1)
        std = self.std_[None, :, None]
        return (X - mean) / std
    

class ECGAugmenter(BaseEstimator, TransformerMixin):
    """
    Prosta augmentacja dla sygnału EKG:
    - losowy Gaussian noise
    - losowe skalowanie amplitudy
    - losowe przesunięcie w czasie

    Używaj TYLKO na zbiorze treningowym.
    """

    def __init__(
        self,
        p_noise=0.5,
        noise_std=0.01,
        p_scale=0.5,
        scale_range=(0.9, 1.1),
        p_shift=0.5,
        max_shift=50,
        random_state=None,
    ):
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_scale = p_scale
        self.scale_min, self.scale_max = scale_range
        self.p_shift = p_shift
        self.max_shift = max_shift
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        return self

    def _time_shift(self, x, shift):
        """
        x: (n_leads, n_timesteps)
        dodatni shift -> w prawo, uzupełniamy zerami
        """
        n_leads, n_timesteps = x.shape
        x_shifted = np.zeros_like(x)
        if shift > 0:
            x_shifted[:, shift:] = x[:, :-shift]
        elif shift < 0:
            shift = -shift
            x_shifted[:, :-shift] = x[:, shift:]
        else:
            x_shifted = x
        return x_shifted

    def transform(self, X, y=None):
        X = np.asarray(X)
        X_aug = X.copy()

        n_records, n_leads, n_timesteps = X.shape

        for i in range(n_records):
            xi = X_aug[i]

            # skala
            if np.random.rand() < self.p_scale:
                scale = np.random.uniform(self.scale_min, self.scale_max)
                xi = xi * scale

            # szum
            if np.random.rand() < self.p_noise:
                noise = np.random.normal(0.0, self.noise_std, size=xi.shape)
                xi = xi + noise

            # przesunięcie
            if np.random.rand() < self.p_shift:
                shift = np.random.randint(-self.max_shift, self.max_shift + 1)
                xi = self._time_shift(xi, shift)

            X_aug[i] = xi

        return X_aug
    
class ECGDerivativeAugmenter(BaseEstimator, TransformerMixin):
    """
    Dodaje pochodną po czasie jako dodatkowe kanały.
    Input:  X: (N, L, T)
    Output: X: (N, 2L, T) – oryginalne kanały + ich pochodne.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        # pochodna po czasie (różnice sąsiednich próbek)
        dX = np.diff(X, axis=-1, prepend=X[..., :1])
        # sklejamy po osi leadów
        X_aug = np.concatenate([X, dX], axis=1)
        return X_aug

# Transformery do klasycznego modelu Ml (baseline)

import scipy.stats


class ECGFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformator wyciągający cechy statystyczne z sygnału EKG.
    Input: numpy array (N, L, T)
    Output: numpy array (N, L * liczba_cech)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_records, n_leads, n_samples = X.shape

        all_features = []
        for i in range(n_records):
            record_features = []
            for lead in range(n_leads):
                sig = X[i, lead, :]
                # Podstawowe statystyki per odprowadzenie
                record_features.extend([
                    np.mean(sig),
                    np.std(sig),
                    np.median(sig),
                    np.min(sig),
                    np.max(sig),
                    np.var(sig),
                    scipy.stats.skew(sig),
                    scipy.stats.kurtosis(sig)
                ])
            all_features.append(record_features)

        return np.array(all_features)