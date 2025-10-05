import pandas as pd
import numpy as np
import os

from mlp import MLP
from datetime import datetime
from matplotlib import pyplot as plt

SEED = 654654
N_EPOCHS = 100
DATA_FRACTION = 0.05

FILEPATH = os.path.abspath(__file__)
INPUT_DATA_DIR = os.path.join(os.path.dirname(FILEPATH), '..', 'data', 'processed', 'input', 'sequences')
OUTPUT_DATA_DIR = os.path.join(os.path.dirname(FILEPATH), '..', 'data', 'processed', 'output', 'sequences')

GENERATOR = np.random.default_rng(SEED)
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

patient_ids = [int(f.split(".")[0]) for f in os.listdir(INPUT_DATA_DIR)]
training_patient_ids = GENERATOR.choice(patient_ids, size=2, replace=False)
testing_patient_ids = [pid for pid in patient_ids if pid not in training_patient_ids]

layer_configs = [
    {"size": 69, "activation_function": "sigmoid", "weights_initializer": "xavier-uniform"},
    {"size": 46, "activation_function": "sigmoid", "weights_initializer": "xavier-uniform"},
    {"size": 23, "activation_function": "sigmoid", "weights_initializer": "xavier-uniform"}
]

output_layer_config = {"size": 1, "loss_function": "bce", "activation_function": "sigmoid", "weights_initializer": "xavier-uniform"}

model = MLP(n_inputs=48, output_layer_config=output_layer_config, hidden_layers_config=layer_configs, learning_rate=0.01, seed=42)

dfs = list()
for pid in training_patient_ids:
    df_i = pd.read_parquet(os.path.join(INPUT_DATA_DIR, f"{pid}.parquet"))
    df_o = pd.read_parquet(os.path.join(OUTPUT_DATA_DIR, f"{pid}.parquet"))
    df_i = df_i.sample(frac=DATA_FRACTION, random_state=SEED).reset_index(drop=True)
    df = pd.merge(df_i, df_o, on=["sequence", "file_index"], how="inner")
    df = df.drop(columns=["sequence", "file_index"])
    dfs.append(df)
df_train = pd.concat(dfs, axis=0, ignore_index=True)

dfs = list()
for pid in testing_patient_ids:
    df_i = pd.read_parquet(os.path.join(INPUT_DATA_DIR, f"{pid}.parquet"))
    df_o = pd.read_parquet(os.path.join(OUTPUT_DATA_DIR, f"{pid}.parquet"))
    df_i = df_i.sample(frac=DATA_FRACTION, random_state=SEED).reset_index(drop=True)
    df = pd.merge(df_i, df_o, on=["sequence", "file_index"], how="inner")
    df = df.drop(columns=["sequence", "file_index"])
    dfs.append(df)
df_test = pd.concat(dfs, axis=0, ignore_index=True)

df_train.to_parquet(os.path.join(os.path.dirname(FILEPATH), '..', 'models', f'training_data_mlp_{NOW}.parquet'))
df_test.to_parquet(os.path.join(os.path.dirname(FILEPATH), '..', 'models', f'testing_data_mlp_{NOW}.parquet'))

print(f"Training patients: {training_patient_ids}")
print(f"Testing patients: {testing_patient_ids}")
print(f"Training data shape: {df_train.shape}")

df_train_input = df_train.drop(columns=["snored"])
df_train_input_mean = df_train_input.mean(axis=0)
df_train_input_std = df_train_input.std(axis=0)
df_train_input = (df_train_input - df_train_input_mean) / df_train_input_std
df_train_output = df_train[["snored"]]

X_train = df_train_input.to_numpy()
y_train = df_train_output["snored"].to_numpy()[:, np.newaxis]

model.train(X_train, y_train, epochs=N_EPOCHS)
model.save(os.path.join(os.path.dirname(FILEPATH), '..', 'models', f'mlp_{NOW}.npz'))