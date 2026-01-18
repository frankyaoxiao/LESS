#!/usr/bin/env python3
"""
Convert DeepSpeed ZeRO-sharded optimizer states to LESS-compatible format.

DeepSpeed ZeRO shards optimizer states across GPUs. This script:
1. Loads all shards and concatenates exp_avg and exp_avg_sq
2. Maps the flat vectors back to per-parameter states using adapter weights
3. Saves in the format LESS expects: {"state": {param_name: {"exp_avg": ..., "exp_avg_sq": ...}}}

Usage:
    python convert_deepspeed_optimizer.py /path/to/adapter_dir

The adapter_dir should contain:
    - adapter_model.safetensors (or adapter_model.bin)
    - training_state/pytorch_model/bf16_zero_pp_rank_*_optim_states.pt

Output:
    - optimizer.bin in the adapter_dir
"""

import argparse
import glob
import json
import os
import re

import torch
from safetensors import safe_open


def load_deepspeed_shards(training_state_dir: str):
    """Load and concatenate all DeepSpeed optimizer shards."""

    # Find all optimizer state shards
    pattern = os.path.join(training_state_dir, "bf16_zero_pp_rank_*_optim_states.pt")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(f"No optimizer shards found matching {pattern}")

    print(f"Found {len(shard_files)} optimizer shards")

    all_exp_avg = []
    all_exp_avg_sq = []

    for shard_file in shard_files:
        print(f"  Loading {os.path.basename(shard_file)}...")
        shard = torch.load(shard_file, map_location="cpu", weights_only=False)

        # Navigate to the optimizer state (nested structure)
        state = shard["optimizer_state_dict"]["optimizer_state_dict"]["state"][0]
        all_exp_avg.append(state["exp_avg"])
        all_exp_avg_sq.append(state["exp_avg_sq"])

    # Concatenate all shards
    full_exp_avg = torch.cat(all_exp_avg)
    full_exp_avg_sq = torch.cat(all_exp_avg_sq)

    print(f"Combined shape: {full_exp_avg.shape}")
    return full_exp_avg, full_exp_avg_sq


def load_adapter_params(adapter_dir: str):
    """Load adapter parameter names and shapes."""

    # Try safetensors first, then bin
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    param_names = []
    param_shapes = []

    if os.path.exists(safetensors_path):
        print(f"Loading adapter from {safetensors_path}")
        with safe_open(safetensors_path, framework="pt") as f:
            for key in sorted(f.keys()):
                tensor = f.get_tensor(key)
                param_names.append(key)
                param_shapes.append(tensor.shape)
    elif os.path.exists(bin_path):
        print(f"Loading adapter from {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
        for key in sorted(state_dict.keys()):
            param_names.append(key)
            param_shapes.append(state_dict[key].shape)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    print(f"Found {len(param_names)} parameters")
    return param_names, param_shapes


def reconstruct_optimizer_state(full_exp_avg, full_exp_avg_sq, param_names, param_shapes):
    """Split flat vectors into per-parameter optimizer states."""

    optimizer_state = {}
    offset = 0

    for name, shape in zip(param_names, param_shapes):
        numel = torch.tensor(shape).prod().item()

        exp_avg = full_exp_avg[offset:offset + numel].reshape(shape)
        exp_avg_sq = full_exp_avg_sq[offset:offset + numel].reshape(shape)

        # Fix parameter names: add .default. before .weight/.bias for PEFT compatibility
        if name.endswith('.weight'):
            fixed_name = name.replace('.weight', '.default.weight')
        elif name.endswith('.bias'):
            fixed_name = name.replace('.bias', '.default.bias')
        else:
            fixed_name = name

        optimizer_state[fixed_name] = {
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
        offset += numel

    # Verify we used all elements
    if offset != full_exp_avg.numel():
        print(f"WARNING: Used {offset} elements but had {full_exp_avg.numel()}")

    return optimizer_state


def fix_adapter_config(adapter_dir: str):
    """Remove unsupported fields from adapter_config.json for older PEFT versions."""

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        config = json.load(f)

    # Essential fields for PEFT 0.7.x
    essential_fields = {
        'peft_type', 'base_model_name_or_path', 'task_type',
        'r', 'lora_alpha', 'lora_dropout', 'target_modules',
        'bias', 'fan_in_fan_out', 'modules_to_save',
        'init_lora_weights', 'layers_to_transform', 'layers_pattern',
        'rank_pattern', 'alpha_pattern', 'inference_mode'
    }

    removed = [k for k in config.keys() if k not in essential_fields]
    if removed:
        print(f"Removing unsupported config fields: {removed}")
        config = {k: v for k, v in config.items() if k in essential_fields}

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed optimizer to LESS format")
    parser.add_argument("adapter_dir", help="Path to adapter directory")
    parser.add_argument("--training-state-dir", default=None,
                        help="Path to training_state/pytorch_model (default: adapter_dir/training_state/pytorch_model)")
    parser.add_argument("--fix-config", action="store_true",
                        help="Also fix adapter_config.json for older PEFT versions")
    args = parser.parse_args()

    adapter_dir = args.adapter_dir
    training_state_dir = args.training_state_dir or os.path.join(adapter_dir, "training_state", "pytorch_model")

    print("=" * 60)
    print("Converting DeepSpeed Optimizer to LESS Format")
    print("=" * 60)
    print(f"Adapter dir: {adapter_dir}")
    print(f"Training state dir: {training_state_dir}")

    # Step 1: Load DeepSpeed shards
    print("\n[1/4] Loading DeepSpeed optimizer shards...")
    full_exp_avg, full_exp_avg_sq = load_deepspeed_shards(training_state_dir)

    # Step 2: Load adapter parameters
    print("\n[2/4] Loading adapter parameters...")
    param_names, param_shapes = load_adapter_params(adapter_dir)

    # Verify sizes match
    total_params = sum(torch.tensor(s).prod().item() for s in param_shapes)
    if total_params != full_exp_avg.numel():
        raise ValueError(f"Size mismatch: adapter has {total_params} params, optimizer has {full_exp_avg.numel()}")

    # Step 3: Reconstruct per-parameter states
    print("\n[3/4] Reconstructing per-parameter optimizer states...")
    optimizer_state = reconstruct_optimizer_state(full_exp_avg, full_exp_avg_sq, param_names, param_shapes)

    # Step 4: Save
    print("\n[4/4] Saving optimizer.bin...")
    output_path = os.path.join(adapter_dir, "optimizer.bin")
    torch.save({"state": optimizer_state}, output_path)
    print(f"Saved to {output_path}")

    # Optionally fix adapter config
    if args.fix_config:
        print("\nFixing adapter_config.json...")
        fix_adapter_config(adapter_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Parameters: {len(optimizer_state)}")
    print(f"  Total elements: {full_exp_avg.numel():,}")
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
