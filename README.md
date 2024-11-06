# Lipophilicity Model Training

This repository contains a script to train a model on the Lipophilicity dataset using Morgan fingerprints and MACCS keys.

## Requirements
Use the provided `environment.yml` file to create a conda environment.

## Usage

Navigate to the `scripts/` directory and run the following command:

```bash
cd scripts/
conda activate lipophilicity_model
python train_model.py --hidden_layers 100 50 --max_iter 1000

