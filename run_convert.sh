#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate less
cd /mnt/polished-lake/home/fxiao-two/LESS
python3 less/scripts/convert_deepspeed_optimizer.py \
    checkpoints/olmo2-dpo-lora \
    --training-state-dir /mnt/polished-lake/home/fxiao-two/openinstruct/output/lora/training_state/pytorch_model \
    --fix-config
