import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from custom_transformers import LabelBuilder

RANDOM_STATE = 42
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

db = pd.read_csv(DATABASE_PATH)
st = pd.read_csv(STATEMENTS_PATH)

label_builder = LabelBuilder(st_df=st)
db_labeled = label_builder.transform(db)


def custom_patient_split(df, val_prop=0.15, test_prop=0.15, random_state=42):
    rng = np.random.default_rng(random_state)
    total_n = len(df)

    #statystyki po pacjencie
    patient_stats = df.groupby("patient_id")["validated_by_human"].agg(
        n_rows="count",
        min_val="min",
        max_val="max"
    )
    # all_valid = min == max == 1 
    patient_stats["all_validated"] = (patient_stats["min_val"] == 1)

    # A: tylko pacjenci, gdzie każde badanie ma validated_by_human == 1
    A = patient_stats[patient_stats["all_validated"]].copy()
    # B: reszta
    B = patient_stats[~patient_stats["all_validated"]].copy()

    target_val_test = int(round((val_prop + test_prop) * total_n))
    target_val = int(round(val_prop * total_n))

    patients_A = A.index.to_numpy()
    rng.shuffle(patients_A)

    S = []
    count_S = 0
    for pid in patients_A:
        n_rows = int(A.loc[pid, "n_rows"])
        if count_S + n_rows <= target_val_test or len(S) == 0:
            S.append(pid)
            count_S += n_rows
        else:
            break

    S = np.array(S)
    rng.shuffle(S)

    val_patients = []
    test_patients = []

    val_count = 0
    target_val_exact = target_val

    for pid in S:
        n_rows = int(A.loc[pid, "n_rows"])
        if val_count + n_rows <= target_val_exact or len(val_patients) == 0:
            val_patients.append(pid)
            val_count += n_rows
        else:
            test_patients.append(pid)

    val_patients = set(val_patients)
    test_patients = set(test_patients)
    S_set = set(S)

    all_patients = set(patient_stats.index)
    train_patients = (all_patients - S_set)

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df   = df[df["patient_id"].isin(val_patients)].copy()
    test_df  = df[df["patient_id"].isin(test_patients)].copy()

    assert len(set(train_df.patient_id) & set(val_df.patient_id)) == 0
    assert len(set(train_df.patient_id) & set(test_df.patient_id)) == 0
    assert len(set(val_df.patient_id) & set(test_df.patient_id)) == 0

    assert val_df["validated_by_human"].min() == 1
    assert test_df["validated_by_human"].min() == 1

    print("Procenty danych zwalidowanych przez czlowieka w kazdym ze zbiorow")
    print("val:",val_df["validated_by_human"].mean())
    print("test:",test_df["validated_by_human"].mean())
    print("train:",train_df["validated_by_human"].mean())

    #info diagnostyczne
    print(f"Total rows: {total_n}")
    print(f"Train rows: {len(train_df)} ({len(train_df)/total_n:.2%})")
    print(f"Val   rows: {len(val_df)} ({len(val_df)/total_n:.2%})")
    print(f"Test  rows: {len(test_df)} ({len(test_df)/total_n:.2%})")

    print("\nRozkład etykiet:")
    for name, d in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        print(f"\n{name}:")
        print(d["label"].value_counts(normalize=True).round(3))

    return train_df, val_df, test_df

train_df, val_df, test_df = custom_patient_split(db_labeled, val_prop=0.15, test_prop=0.15, random_state=42)


#zapisanie do csv
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

train_path = os.path.join(ARTIFACTS_DIR, "ptbxl_train.csv")
val_path   = os.path.join(ARTIFACTS_DIR, "ptbxl_val.csv")
test_path  = os.path.join(ARTIFACTS_DIR, "ptbxl_test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nSplit zapisany do CSV:")
print(" -", train_path)
print(" -", val_path)
print(" -", test_path)
