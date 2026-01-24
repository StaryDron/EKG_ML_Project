import pandas as pd
import os
from data_ptbxl import make_dataloaders

RANDOM_STATE = 42
BASE_DIR = os.getcwd()

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

ART_DIR = os.path.join(PROJECT_ROOT_DIR, "artifacts")

TRAIN_PATH= os.path.join(ART_DIR,"ptbxl_train.csv")
PTB_DIR = os.path.join(PROJECT_ROOT_DIR, "PTB-XL")

VAL_PATH  = os.path.join(ART_DIR, "ptbxl_val.csv")
TEST_PATH = os.path.join(ART_DIR, "ptbxl_test.csv")

val_df  = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH)

train_loader, val_loader, test_loader, pipe, info = make_dataloaders(
    train_df, val_df, test_df,
    ptb_dir=PTB_DIR,
    batch_size=64,
    num_workers=0,
    use_derivative=False,
    use_augment=False
)

print("INFO:", info)

x, y = next(iter(train_loader))
print("BATCH x:", x.shape, x.dtype)
print("BATCH y:", y.shape, y.dtype)



