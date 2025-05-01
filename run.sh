#!/bin/bash

ACTUAL_CACHE=/import/ml-sc-scratch6/lang/cache/

export HF_DATASET_CACHE=${ACTUAL_CACHE}
export HF_MODULES_CACHE=${ACTUAL_CACHE}
export HF_HOME=${ACTUAL_CACHE}

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama_finetuning_on_llava_med_gpu.yaml > /import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu/train_output.log 2>&1
