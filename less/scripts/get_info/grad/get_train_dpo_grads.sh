#!/bin/bash

# Collect DPO training gradients with Adam optimizer state normalization

dpo_train_file=$1  # path to DPO training data (JSONL with chosen/rejected pairs)
model=$2           # path to LoRA model checkpoint (must have adapter_config.json and optimizer.bin)
output_path=$3     # path to output
dims=$4            # dimension of projection, can be a list (e.g., "8192" or "4096 8192")
gradient_type=$5   # adam or sgd (default: adam for training)
dpo_beta=${6:-5.0} # DPO temperature parameter (default: 5.0)

if [[ -z "$gradient_type" ]]; then
    gradient_type="adam"
fi

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

echo "Collecting DPO training gradients..."
echo "  DPO file: $dpo_train_file"
echo "  Model: $model"
echo "  Output: $output_path"
echo "  Dims: $dims"
echo "  Gradient type: $gradient_type"
echo "  DPO beta: $dpo_beta"

python3 -m less.data_selection.get_info \
    --dpo_train_file $dpo_train_file \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type $gradient_type \
    --loss_type dpo \
    --dpo_beta $dpo_beta
