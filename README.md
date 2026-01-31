# ECG Classification on PTB-XL using Deep Learning

##  Project Overview

This project focuses on **automatic classification of ECG signals** from the **PTB-XL dataset** using deep learning models.
We build a **reproducible pipeline** including preprocessing, feature engineering (time derivatives), data augmentation, and multiple neural network architectures (ResNet1D, Inception-style models).

The task is **multiclass classification** into 5 diagnostic superclasses:
- **NORM** – Normal ECG
- **MI** – Myocardial Infarction
- **STTC** – ST/T Changes
- **HYP** – Hypertrophy
- **CD** – Conduction Disturbance

##  Repository Structure
```
├── artifacts/
│ ├── ptbxl_train.csv
│ ├── ptbxl_val.csv
│ ├── ptbxl_test.csv
│ ├── baseline/
│ ├── resnet_standard/
│ ├── resnet_augmented/
│ ├── inception_standard/
│ ├── inception_augmented/
│ ├── eda_plots/
│ ├── interpretability_results/
│ └── final_results/
│
├── PTB-XL/
│ ├── records100/
│ ├── ptbxl_database.csv
│ └── scp_statements.csv
│
├── codes/
│ ├── custom_transformers.py
│ ├── data_ptbxl.py
│ ├── pipeline.py
│ ├── train_resnet1d.py
│ ├── train_resnet1d_augmented.py
│ ├── inception_model.py
│ ├── train_inception.py
│ ├── train_inception_augmented.py
│ ├── train_baseline_ml.py
│ ├── conf_matrix_and_stat_test.py
│ ├── plots_and_metrics.py
│ ├── interpretability.py
│ └── eda.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```
Structure after downloading PTB-XL
and running all of the codes 

artifacts containes all reources produced and 
used by the code including models and charts.

---

##  How to run the project

unpack the PTB-XL.zip it contains a small demonstrative fraction
of the database(only 100 ekg signals) and make sure that it is in the 
exact position as shown in the repository structure after unpacking and
that it has the same name.

pip install -r requirements.txt
this includes torch 2.5.1+cu121 that makes use of the gpu
if you don't have a gpu, executing this project will take at least 12 hours.

Run  files in the folowing order:

1. eda.ipynb
2. custom_transformers
3. pipeline
4. data_ptbxl
5. train_baseline_ml
6. train_resnet1d
7. train_resnet1d_augmented
8. inception_model
9. train_inception
10. train_inception_augmented
11. plots_and_metrics
12. conf_matrix_and_stat_test
13. interpretability
