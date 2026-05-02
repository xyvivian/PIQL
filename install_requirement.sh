#!/bin/bash

set -e

echo "Creating conda environment..."
conda create -y -n piql python=3.10
source $(conda info --base)/etc/profile.d/conda.sh
conda activate piql

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core ML stack..."

pip3 install torch torchvision

pip install \
    numpy \
    scipy \
    pandas \
    scikit-learn

echo "Installing transformer / foundation model deps..."
pip install \
    transformers \
    datasets \
    huggingface-hub \
    tokenizers \
    einops

echo "Installing OD + research libs..."
pip install \
    pyod \
    adbench \
    copulas \
    configspace

echo "Installing training / utilities..."
pip install \
    pytorch-lightning \
    hydra-core \
    omegaconf \
    tqdm \
    wandb

echo "Optional (remove if not needed): Jupyter + plotting"
pip install \
    matplotlib \
    seaborn \
    jupyterlab

echo "Done. Activate with: conda activate piql"
