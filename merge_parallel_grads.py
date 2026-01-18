"""
Merge gradient files from parallel GPU runs into a single output.

Usage:
    python merge_parallel_grads.py [--input_base outputs/olmo_dpo_grads_parallel] [--output_dir outputs/olmo_dpo_grads_merged]
"""

import argparse
import os
import torch

def merge_parallel_outputs(input_base: str, output_dir: str, num_gpus: int = 8, dim: int = 8192):
    """
    Merge gradient outputs from parallel GPU runs.

    Args:
        input_base: Base directory containing gpu0/, gpu1/, etc.
        output_dir: Output directory for merged gradients
        num_gpus: Number of GPUs used
        dim: Projection dimension
    """
    print(f"Merging outputs from {num_gpus} GPUs...")
    print(f"Input base: {input_base}")
    print(f"Output dir: {output_dir}")
    print()

    # Create output directory
    output_dim_dir = os.path.join(output_dir, f"dim{dim}")
    os.makedirs(output_dim_dir, exist_ok=True)

    # Collect all gradients in order
    all_grads = []
    all_grads_unnorm = []
    total_samples = 0

    for gpu_id in range(num_gpus):
        gpu_dir = os.path.join(input_base, f"gpu{gpu_id}", f"dim{dim}")

        # Load normalized gradients
        norm_path = os.path.join(gpu_dir, "all_orig.pt")
        if not os.path.exists(norm_path):
            print(f"WARNING: Missing {norm_path}")
            continue

        grads = torch.load(norm_path)
        all_grads.append(grads)
        print(f"GPU {gpu_id}: {grads.shape[0]} samples (normalized)")
        total_samples += grads.shape[0]

        # Load unnormalized gradients if available
        unnorm_path = os.path.join(gpu_dir, "all_unormalized.pt")
        if os.path.exists(unnorm_path):
            grads_unnorm = torch.load(unnorm_path)
            all_grads_unnorm.append(grads_unnorm)

    print()
    print(f"Total samples: {total_samples}")

    # Concatenate all gradients
    merged_grads = torch.cat(all_grads, dim=0)
    print(f"Merged normalized shape: {merged_grads.shape}")

    # Save merged normalized gradients
    output_path = os.path.join(output_dim_dir, "all_orig.pt")
    torch.save(merged_grads, output_path)
    print(f"Saved normalized gradients to: {output_path}")

    # Save merged unnormalized gradients if available
    if len(all_grads_unnorm) == num_gpus:
        merged_unnorm = torch.cat(all_grads_unnorm, dim=0)
        unnorm_output_path = os.path.join(output_dim_dir, "all_unormalized.pt")
        torch.save(merged_unnorm, unnorm_output_path)
        print(f"Saved unnormalized gradients to: {unnorm_output_path}")

    print()
    print("Merge complete!")

    # Verify merged output
    print()
    print("=== Verification ===")
    print(f"Shape: {merged_grads.shape}")
    print(f"Dtype: {merged_grads.dtype}")
    print(f"NaN count: {torch.isnan(merged_grads).sum().item()}")
    print(f"Inf count: {torch.isinf(merged_grads).sum().item()}")

    # Check row norms
    row_norms = merged_grads.float().norm(dim=1)
    valid_norms = ((row_norms > 0.99) & (row_norms < 1.01)).sum().item()
    print(f"Valid row norms (0.99-1.01): {valid_norms}/{len(row_norms)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base", type=str, default="outputs/olmo_dpo_grads_parallel",
                        help="Base directory with parallel outputs")
    parser.add_argument("--output_dir", type=str, default="outputs/olmo_dpo_grads_merged",
                        help="Output directory for merged gradients")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs used")
    parser.add_argument("--dim", type=int, default=8192, help="Projection dimension")
    args = parser.parse_args()

    merge_parallel_outputs(args.input_base, args.output_dir, args.num_gpus, args.dim)
