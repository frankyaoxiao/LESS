#!/bin/bash
# Parallel DPO gradient collection across 8 GPUs

TOTAL_SAMPLES=378341
NUM_GPUS=8
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS + 1))  # 47293

OUTPUT_BASE="outputs/olmo_dpo_grads_parallel"
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Total samples: $TOTAL_SAMPLES"
echo "GPUs: $NUM_GPUS"
echo "Samples per GPU: $SAMPLES_PER_GPU"
echo ""

# Launch jobs on each GPU
for i in $(seq 0 $((NUM_GPUS - 1))); do
    START_IDX=$((i * SAMPLES_PER_GPU))
    OUTPUT_PATH="${OUTPUT_BASE}/gpu${i}"
    LOG_FILE="${LOG_DIR}/gpu${i}.log"

    echo "GPU $i: start_index=$START_IDX, max_samples=$SAMPLES_PER_GPU"

    CUDA_VISIBLE_DEVICES=$i python -m less.data_selection.get_info \
        --model_path checkpoints/olmo2-dpo-lora \
        --dpo_train_file data/olmo_dpo_train.jsonl \
        --output_path $OUTPUT_PATH \
        --info_type grads \
        --loss_type dpo \
        --dpo_beta 5.0 \
        --gradient_type adam \
        --gradient_projection_dimension 8192 \
        --start_index $START_IDX \
        --max_samples $SAMPLES_PER_GPU \
        > $LOG_FILE 2>&1 &

    echo "  Launched (PID: $!), logging to $LOG_FILE"
done

echo ""
echo "All jobs launched. Monitor with:"
echo "  tail -f logs/gpu*.log"
echo ""
echo "When complete, run:"
echo "  python merge_parallel_grads.py"
