import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from custom_transformers import (
    ECGSignalLoader,
    ECGBandpassFilter,
    ECGAmplitudeClipper,
    ECGFeatureExtractor
)

def main():
    BASE_DIR = os.getcwd()
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    PTB_DIR = os.path.join(PROJECT_ROOT_DIR, "PTB-XL")
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT_DIR, "artifacts")

    train_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "ptbxl_train.csv"))
    val_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "ptbxl_val.csv"))
    test_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "ptbxl_test.csv"))

    LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]
    label2id = {lab: i for i, lab in enumerate(LABELS)}

    y_train = train_df["label"].map(label2id).to_numpy()
    y_val = val_df["label"].map(label2id).to_numpy()
    y_test = test_df["label"].map(label2id).to_numpy()

    data_pipeline = Pipeline([
        ('loader', ECGSignalLoader(ptb_dir=PTB_DIR, filename_col="filename_lr")),
        ('filter', ECGBandpassFilter(lowcut=0.5, highcut=40.0, fs=100.0)),
        ('clipper', ECGAmplitudeClipper(k=5.0)),
        ('features', ECGFeatureExtractor())
    ])

    X_train_feats = data_pipeline.fit_transform(train_df)
    X_val_feats = data_pipeline.transform(val_df)
    X_test_feats = data_pipeline.transform(test_df)

    #zastępujemy NaN średnią z danej cechy wyliczoną na zbiorze treningowym
    imputer = SimpleImputer(strategy='mean')
    X_train_feats = imputer.fit_transform(X_train_feats)
    X_val_feats = imputer.transform(X_val_feats)
    X_test_feats = imputer.transform(X_test_feats)

    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_feats)
    X_val_final = scaler.transform(X_val_feats)
    X_test_final = scaler.transform(X_test_feats)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train_final, y_train)

    def evaluate(X, y, set_name):
        y_prob = rf.predict_proba(X)  
        auc = roc_auc_score(y, y_prob, multi_class='ovr', average='macro')

        y_pred = rf.predict(X)
        acc = np.mean(y_pred == y)

        print(f"\n--- WYNIKI: {set_name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro AUC-ROC: {auc:.4f}")
        return auc

    evaluate(X_val_final, y_val, "WALIDACYJNY")
    evaluate(X_test_final, y_test, "TESTOWY")

    y_pred_test = rf.predict(X_test_final)
    print("\nSzczegółowy raport (Test):")
    print(classification_report(y_test, y_pred_test, target_names=LABELS))

    BASELINE_DIR = os.path.join(ARTIFACTS_DIR, "baseline")
    os.makedirs(BASELINE_DIR, exist_ok=True)

    joblib.dump(rf, os.path.join(BASELINE_DIR, "baseline_rf_model.pkl"))
    joblib.dump(scaler, os.path.join(BASELINE_DIR, "baseline_scaler.pkl"))

    np.save(os.path.join(BASELINE_DIR, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(BASELINE_DIR, "X_val_final.npy"), X_val_final)
    np.save(os.path.join(BASELINE_DIR, "X_test_final.npy"), X_test_final)

    np.save(os.path.join(BASELINE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(BASELINE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(BASELINE_DIR, "y_test.npy"), y_test)

if __name__ == "__main__":
    main()
