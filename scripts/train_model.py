#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


# In[ ]:


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model on Morgan fingerprints or MACCS keys")
parser.add_argument("--hidden_layers", type=int, nargs='+', default=[100, 50],
                    help="Hidden layer sizes for MLP (e.g., --hidden_layers 100 50)")
parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations for MLP")
args = parser.parse_args()

# Load dataset
df = pd.read_csv('../data/Lipophilicity.csv')


# In[ ]:


# Function to generate Morgan fingerprints
def morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return np.zeros((n_bits,))

# Function to generate MACCS keys
def maccs_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return MACCSkeys.GenMACCSKeys(mol)
    else:
        return np.zeros((166,))


# In[ ]:


# Apply functions to generate fingerprints
df['morgan_fp'] = df['smiles'].apply(morgan_fingerprint)
df['maccs_keys'] = df['smiles'].apply(maccs_keys)

# Convert fingerprints to numpy arrays
morgan_fps = np.array(list(df['morgan_fp'].values))
maccs_keys = np.array(list(df['maccs_keys'].values))
y = df['exp'].values

# Split data into train and test sets
X_train_morgan, X_test_morgan, y_train, y_test = train_test_split(morgan_fps, y, test_size=0.2, random_state=42)
X_train_maccs, X_test_maccs, _, _ = train_test_split(maccs_keys, y, test_size=0.2, random_state=42)

# Scale the target
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()


# In[ ]:


# Train MLP model on Morgan fingerprints
mlp_morgan = MLPRegressor(hidden_layer_sizes=tuple(args.hidden_layers), max_iter=args.max_iter, random_state=42)
mlp_morgan.fit(X_train_morgan, y_train_scaled)

# Predict and compute RMSE for Morgan fingerprints
y_pred_morgan_scaled = mlp_morgan.predict(X_test_morgan)
y_pred_morgan = scaler.inverse_transform(y_pred_morgan_scaled.reshape(-1, 1)).ravel()
rmse_morgan = np.sqrt(mean_squared_error(y_test, y_pred_morgan))

# Retrieve conda environment name
conda_env = os.getenv("CONDA_DEFAULT_ENV", "base")

# Save results
with open("../results/results.txt", "w") as file:
    file.write(f"RMSE for Morgan fingerprints: {rmse_morgan}\n")
    file.write(f"Conda environment: {conda_env}\n")
    file.write(f"Hyperparameters: hidden_layers={args.hidden_layers}, max_iter={args.max_iter}\n")

print(f"RMSE for Morgan fingerprints: {rmse_morgan}")
print(f"Results saved to results.txt")

