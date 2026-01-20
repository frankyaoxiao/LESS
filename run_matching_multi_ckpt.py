"""
Multi-checkpoint DPO influence score computation and data selection.

Combines influence scores across multiple checkpoints using weighted averaging,
following the original LESS methodology.
"""

import argparse
import json
import torch
import os
from tqdm import tqdm


def extract_prompt(messages):
    """Extract the user prompt from a message list."""
    if isinstance(messages, str):
        messages = json.loads(messages)
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def compute_influence_for_checkpoint(train_grads_path, valid_grads_path, batch_size, device):
    """Compute influence scores for a single checkpoint."""
    # Load gradients
    train_grads = torch.load(train_grads_path, map_location='cpu')
    valid_grads = torch.load(valid_grads_path, map_location='cpu')

    print(f"  Train grads shape: {train_grads.shape}")
    print(f"  Valid grads shape: {valid_grads.shape}")

    # Find valid (non-NaN, non-zero) training samples
    train_norms = torch.norm(train_grads, dim=1)
    train_valid_mask = ~torch.isnan(train_norms) & (train_norms > 0)

    n_total = len(train_grads)
    n_valid = train_valid_mask.sum().item()
    print(f"  Valid training samples: {n_valid}/{n_total}")

    # Move validation gradients to device
    valid_grads = valid_grads.to(device).float()

    # Initialize scores (invalid ones get NaN to track across checkpoints)
    influence_scores = torch.full((n_total, valid_grads.shape[0]), float('nan'), dtype=torch.float32)

    # Compute influence scores in batches (only for valid samples)
    valid_indices = train_valid_mask.nonzero().squeeze().tolist()
    if isinstance(valid_indices, int):
        valid_indices = [valid_indices]

    for i in tqdm(range(0, len(valid_indices), batch_size), desc="  Computing", leave=False):
        batch_end = min(i + batch_size, len(valid_indices))
        batch_indices = valid_indices[i:batch_end]

        # Get batch of training gradients
        batch_train = train_grads[batch_indices].to(device).float()

        # Compute influence: [batch, dim] @ [dim, n_valid] = [batch, n_valid]
        batch_scores = torch.matmul(batch_train, valid_grads.T)

        influence_scores[batch_indices] = batch_scores.cpu()

    return influence_scores, train_valid_mask


def main():
    parser = argparse.ArgumentParser(description='Multi-checkpoint DPO influence scores')
    parser.add_argument('--train_grads_pattern', type=str, required=True,
                        help='Pattern for training gradients, e.g., outputs/grads_epoch_{}/all_orig.pt')
    parser.add_argument('--valid_grads_pattern', type=str, required=True,
                        help='Pattern for validation gradients, e.g., outputs/valid_grads_epoch_{}/all_orig.pt')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Checkpoint indices to use (default: 0 1 2 3)')
    parser.add_argument('--checkpoint_weights', type=float, nargs='+', default=None,
                        help='Weights for each checkpoint (default: equal weights)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to original training data JSONL')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save outputs')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for matrix multiplication')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum', 'max'],
                        help='How to aggregate scores across validation samples')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalize checkpoint weights
    if args.checkpoint_weights is None:
        args.checkpoint_weights = [1.0 / len(args.checkpoints)] * len(args.checkpoints)
    else:
        assert len(args.checkpoint_weights) == len(args.checkpoints)
        total = sum(args.checkpoint_weights)
        args.checkpoint_weights = [w / total for w in args.checkpoint_weights]

    print(f"Checkpoints: {args.checkpoints}")
    print(f"Weights: {args.checkpoint_weights}")

    # Compute influence scores for each checkpoint
    combined_scores = None
    combined_valid_mask = None

    for ckpt_idx, weight in zip(args.checkpoints, args.checkpoint_weights):
        train_path = args.train_grads_pattern.format(ckpt_idx)
        valid_path = args.valid_grads_pattern.format(ckpt_idx)

        print(f"\nProcessing checkpoint {ckpt_idx} (weight={weight:.3f})...")
        print(f"  Train grads: {train_path}")
        print(f"  Valid grads: {valid_path}")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training gradients not found: {train_path}")
        if not os.path.exists(valid_path):
            raise FileNotFoundError(f"Validation gradients not found: {valid_path}")

        scores, valid_mask = compute_influence_for_checkpoint(
            train_path, valid_path, args.batch_size, device
        )

        # Initialize or accumulate
        if combined_scores is None:
            combined_scores = weight * scores
            combined_valid_mask = valid_mask
        else:
            # Weight and add (NaN handling: if either is NaN, result is NaN)
            combined_scores += weight * scores
            combined_valid_mask = combined_valid_mask & valid_mask

    print(f"\nCombined scores shape: {combined_scores.shape}")
    print(f"Valid across all checkpoints: {combined_valid_mask.sum().item()}")

    # Aggregate across validation samples
    print(f"\nAggregating scores ({args.aggregation})...")
    if args.aggregation == 'mean':
        aggregated_scores = torch.nanmean(combined_scores, dim=1)
    elif args.aggregation == 'sum':
        aggregated_scores = torch.nansum(combined_scores, dim=1)
    elif args.aggregation == 'max':
        aggregated_scores = torch.nan_to_num(combined_scores, nan=float('-inf')).max(dim=1)[0]

    # Replace NaN with -inf for sorting
    aggregated_scores = torch.where(
        torch.isnan(aggregated_scores),
        torch.full_like(aggregated_scores, float('-inf')),
        aggregated_scores
    )

    # Sort by aggregated score (descending)
    print("Sorting by influence score...")
    sorted_scores, sorted_indices = torch.sort(aggregated_scores, descending=True)

    # Filter out invalid (keep only finite scores)
    valid_mask = torch.isfinite(sorted_scores)
    sorted_scores = sorted_scores[valid_mask]
    sorted_indices = sorted_indices[valid_mask]

    print(f"  Sorted {len(sorted_scores)} valid samples")
    print(f"  Top score: {sorted_scores[0].item():.6f}")
    print(f"  Bottom score: {sorted_scores[-1].item():.6f}")
    print(f"  Mean score: {sorted_scores.mean().item():.6f}")

    # Load original training data
    print(f"\nLoading original training data from {args.train_data}...")
    with open(args.train_data, 'r') as f:
        train_lines = f.readlines()
    print(f"  Loaded {len(train_lines)} lines")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Write sorted output
    output_file = os.path.join(args.output_path, "influence_sorted.jsonl")
    print(f"\nWriting sorted data to {output_file}...")

    with open(output_file, 'w') as f:
        for rank, (score, orig_idx) in enumerate(tqdm(
            zip(sorted_scores.tolist(), sorted_indices.tolist()),
            total=len(sorted_scores),
            desc="Writing"
        )):
            # Load original data
            orig_data = json.loads(train_lines[orig_idx])

            # Extract prompt from chosen messages
            chosen = orig_data.get('chosen', [])
            rejected = orig_data.get('rejected', [])
            prompt = extract_prompt(chosen)

            # Build output record
            output_record = {
                "uid": str(orig_idx),
                "prompt": prompt,
                "chosen": json.dumps(chosen) if isinstance(chosen, list) else chosen,
                "rejected": json.dumps(rejected) if isinstance(rejected, list) else rejected,
                "influence_score": round(score, 6),
                "rank": rank + 1,
            }

            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    print(f"\nDone! Output saved to {output_file}")

    # Also save raw scores for analysis
    scores_file = os.path.join(args.output_path, "influence_scores.pt")
    torch.save({
        'aggregated_scores': aggregated_scores,
        'sorted_indices': sorted_indices,
        'sorted_scores': sorted_scores,
        'checkpoints': args.checkpoints,
        'checkpoint_weights': args.checkpoint_weights,
    }, scores_file)
    print(f"Saved raw scores to {scores_file}")

    # Print summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Checkpoints used: {args.checkpoints}")
    print(f"  Checkpoint weights: {args.checkpoint_weights}")
    print(f"  Total training samples: {len(train_lines)}")
    print(f"  Valid samples: {len(sorted_scores)}")
    print(f"  Output file: {output_file}")
    print(f"  Top 5 influential samples (by original index):")
    for i in range(min(5, len(sorted_indices))):
        print(f"    #{i+1}: idx={sorted_indices[i].item()}, score={sorted_scores[i].item():.4f}")


if __name__ == "__main__":
    main()
