#!/bin/bash
# Multi-checkpoint LESS workflow for DPO data attribution
# Run this on the 8-GPU machine

set -e

# Configuration
EPOCHS="0 1 2 3"
NUM_GPUS=8
DPO_BETA=5.0
GRAD_DIM=8192
TRAIN_DATA="data/olmo_dpo_train.jsonl"
VALID_DATA="data/steer_150_preference.jsonl"

# Calculate samples per GPU (378341 total training samples)
TOTAL_SAMPLES=378341
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))  # 47292

echo "=============================================="
echo "Multi-checkpoint LESS Workflow"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "GPUs: $NUM_GPUS"
echo "DPO Beta: $DPO_BETA"
echo "Gradient Dimension: $GRAD_DIM"
echo "Total samples: $TOTAL_SAMPLES"
echo "Samples per GPU: $SAMPLES_PER_GPU"
echo ""

# Create logs directory
mkdir -p logs

# ============================================
# STEP 1: Collect training gradients for each epoch
# ============================================
echo "STEP 1: Collecting training gradients..."

for epoch in $EPOCHS; do
    echo ""
    echo "--- Epoch $epoch: Training gradients ---"

    OUTPUT_DIR="outputs/grads_epoch_${epoch}"
    MERGED_FILE="${OUTPUT_DIR}/dim${GRAD_DIM}/all_orig.pt"

    # Check if already done
    if [ -f "$MERGED_FILE" ]; then
        echo "  Already exists at $MERGED_FILE, skipping..."
        continue
    fi

    # Launch parallel jobs - each GPU processes a slice of the data
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        START_INDEX=$((gpu_id * SAMPLES_PER_GPU))

        # Last GPU gets remaining samples
        if [ $gpu_id -eq $((NUM_GPUS-1)) ]; then
            MAX_SAMPLES=$((TOTAL_SAMPLES - START_INDEX))
        else
            MAX_SAMPLES=$SAMPLES_PER_GPU
        fi

        echo "  Launching GPU $gpu_id: start=$START_INDEX, max=$MAX_SAMPLES"
        CUDA_VISIBLE_DEVICES=$gpu_id python -m less.data_selection.get_info \
            --dpo_train_file $TRAIN_DATA \
            --info_type grads \
            --model_path checkpoints/epoch_${epoch} \
            --output_path ${OUTPUT_DIR}/gpu${gpu_id} \
            --gradient_projection_dimension $GRAD_DIM \
            --gradient_type adam \
            --loss_type dpo \
            --dpo_beta $DPO_BETA \
            --start_index $START_INDEX \
            --max_samples $MAX_SAMPLES > logs/train_epoch_${epoch}_gpu_${gpu_id}.log 2>&1 &
    done

    # Wait for all jobs to complete
    echo "  Waiting for all GPUs to finish..."
    wait
    echo "  Done with epoch $epoch training gradients."

    # Merge immediately
    echo "  Merging gradients..."
    python merge_parallel_grads.py \
        --input_base $OUTPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --num_gpus $NUM_GPUS \
        --dim $GRAD_DIM
done

# ============================================
# STEP 2: Collect validation gradients for each epoch
# ============================================
echo ""
echo "STEP 2: Collecting validation gradients..."

for epoch in $EPOCHS; do
    echo ""
    echo "--- Epoch $epoch: Validation gradients ---"

    OUTPUT_DIR="outputs/valid_grads_epoch_${epoch}"
    MERGED_FILE="${OUTPUT_DIR}/dim${GRAD_DIM}/all_orig.pt"

    if [ -f "$MERGED_FILE" ]; then
        echo "  Already exists at $MERGED_FILE, skipping..."
        continue
    fi

    # Validation is smaller (147 samples), run on single GPU
    CUDA_VISIBLE_DEVICES=0 python -m less.data_selection.get_info \
        --dpo_train_file $VALID_DATA \
        --info_type grads \
        --model_path checkpoints/epoch_${epoch} \
        --output_path ${OUTPUT_DIR}/gpu0 \
        --gradient_projection_dimension $GRAD_DIM \
        --gradient_type sgd \
        --loss_type dpo \
        --dpo_beta $DPO_BETA > logs/valid_epoch_${epoch}.log 2>&1

    # Merge (single GPU, but keeps directory structure consistent)
    python merge_parallel_grads.py \
        --input_base $OUTPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --num_gpus 1 \
        --dim $GRAD_DIM
    echo "  Done."
done

# ============================================
# STEP 3: Run multi-checkpoint matching
# ============================================
echo ""
echo "STEP 3: Running multi-checkpoint matching..."

python run_matching_multi_ckpt.py \
    --train_grads_pattern "outputs/grads_epoch_{}/dim${GRAD_DIM}/all_orig.pt" \
    --valid_grads_pattern "outputs/valid_grads_epoch_{}/dim${GRAD_DIM}/all_orig.pt" \
    --checkpoints 0 1 2 3 \
    --checkpoint_weights 1.0 1.0 1.0 1.0 \
    --train_data $TRAIN_DATA \
    --output_path outputs/influence_multi_ckpt \
    --batch_size 10000 \
    --aggregation mean

echo ""
echo "=============================================="
echo "Done! Results saved to outputs/influence_multi_ckpt/"
echo "=============================================="
