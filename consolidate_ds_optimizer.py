"""
Consolidate DeepSpeed ZeRO sharded optimizer states into a single optimizer.bin file.

This script merges the sharded exp_avg and exp_avg_sq tensors from DeepSpeed checkpoints
and maps them to parameter names for use with LESS gradient collection.
"""

import argparse
import os
import torch
from glob import glob
from tqdm import tqdm
from collections import OrderedDict


def consolidate_optimizer(checkpoint_dir: str, output_path: str, model_path: str = None):
    """
    Consolidate DeepSpeed sharded optimizer states.

    Args:
        checkpoint_dir: Path to training_state/pytorch_model/ directory
        output_path: Where to save consolidated optimizer.bin
        model_path: Path to adapter model (to get parameter names and order)
    """

    # Find all optimizer state files
    optim_files = sorted(glob(os.path.join(checkpoint_dir, "*_optim_states.pt")))
    print(f"Found {len(optim_files)} optimizer state files")

    if len(optim_files) == 0:
        raise ValueError(f"No optimizer state files found in {checkpoint_dir}")

    # Load and concatenate optimizer states
    all_exp_avg = []
    all_exp_avg_sq = []

    for f in tqdm(optim_files, desc="Loading shards"):
        state = torch.load(f, weights_only=False, map_location='cpu')
        inner_state = state['optimizer_state_dict']['optimizer_state_dict']['state']

        # There should be one state entry (key 0)
        for key, val in inner_state.items():
            all_exp_avg.append(val['exp_avg'])
            all_exp_avg_sq.append(val['exp_avg_sq'])

    # Concatenate
    full_exp_avg = torch.cat(all_exp_avg, dim=0)
    full_exp_avg_sq = torch.cat(all_exp_avg_sq, dim=0)

    print(f"Concatenated exp_avg shape: {full_exp_avg.shape}")
    print(f"Concatenated exp_avg_sq shape: {full_exp_avg_sq.shape}")

    # Get parameter names from model if provided
    if model_path:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        print(f"\nLoading model from {model_path} to get parameter names...")

        # Use a known working checkpoint for parameter names
        # (the parameter names and order should be the same across all epochs)
        working_model_path = "checkpoints/olmo2-dpo-lora"
        if os.path.exists(working_model_path):
            print(f"Using {working_model_path} for parameter names (same structure)")
            model_path = working_model_path

        # Try to infer base model from adapter_config
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path) as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "allenai/OLMo-2-1124-7B-SFT")
        else:
            base_model_name = "allenai/OLMo-2-1124-7B-SFT"

        print(f"Base model: {base_model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map='cpu'
        )
        model = PeftModel.from_pretrained(base_model, model_path)

        # Get trainable parameter names in order
        param_names = []
        param_shapes = []
        for name, param in model.named_parameters():
            if 'lora' in name:
                param_names.append(name)
                param_shapes.append(param.shape)

        print(f"Found {len(param_names)} LoRA parameters")

        # Verify total size matches
        total_params = sum(p.numel() for s in param_shapes for p in [torch.zeros(s)])
        total_params = sum(torch.zeros(s).numel() for s in param_shapes)
        print(f"Total LoRA params from model: {total_params}")
        print(f"Total params from optimizer: {full_exp_avg.shape[0]}")

        if total_params != full_exp_avg.shape[0]:
            print("WARNING: Parameter count mismatch!")

        # Split optimizer states by parameter and create state dict
        state_dict = {'state': OrderedDict()}
        offset = 0
        for name, shape in zip(param_names, param_shapes):
            numel = torch.zeros(shape).numel()
            state_dict['state'][name] = {
                'exp_avg': full_exp_avg[offset:offset+numel].view(shape),
                'exp_avg_sq': full_exp_avg_sq[offset:offset+numel].view(shape),
            }
            offset += numel

        print(f"Created state dict with {len(state_dict['state'])} entries")
    else:
        # Just save the flat tensors
        state_dict = {
            'flat_exp_avg': full_exp_avg,
            'flat_exp_avg_sq': full_exp_avg_sq,
        }

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"\nSaved consolidated optimizer to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Consolidate DeepSpeed optimizer states')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to training_state/pytorch_model/ directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to save consolidated optimizer.bin')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to adapter model (to get parameter names)')
    args = parser.parse_args()

    consolidate_optimizer(args.checkpoint_dir, args.output_path, args.model_path)


if __name__ == "__main__":
    main()
