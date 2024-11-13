#!/bin/bash

# Load the conda environment setup script
source /import/ml-sc-scratch5/viekashv/miniconda3/etc/profile.d/conda.sh

# Activate the llama_eval environment
conda activate /import/ml-sc-scratch5/viekashv/miniconda3/envs/llama3_2_finetune

export FORCE_TORCHRUN=1

scl enable devtoolset-9 'llamafactory-cli train /import/ml-sc-scratch5/viekashv/llama3_2_finetuning/LLaMA-Factory/examples/train_full/llama_finetuning_on_adi.yaml'