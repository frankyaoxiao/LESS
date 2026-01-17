#!/bin/bash

# Collect DPO validation gradients (SGD - no optimizer state needed)

dpo_validation_file=$1  # path to DPO validation data (JSONL with chosen/rejected pairs)
model=$2                # path to LoRA model checkpoint
output_path=$3          # path to output
dims=$4                 # dimension of projection, can be a list (e.g., "8192" or "4096 8192")
dpo_beta=${5:-5.0}      # DPO temperature parameter (default: 5.0)

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

echo "Collecting DPO validation gradients..."
echo "  DPO file: $dpo_validation_file"
echo "  Model: $model"
echo "  Output: $output_path"
echo "  Dims: $dims"
echo "  Gradient type: sgd (always for validation)"
echo "  DPO beta: $dpo_beta"

python3 -m less.data_selection.get_info \
    --dpo_validation_file $dpo_validation_file \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type sgd \
    --loss_type dpo \
    --dpo_beta $dpo_beta
