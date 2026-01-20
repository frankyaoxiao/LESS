"""
DPO influence score computation and data selection.

Computes influence scores between training and validation gradients,
aggregates across validation samples, and outputs sorted training data.
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


def main():
    parser = argparse.ArgumentParser(description='Compute DPO influence scores and select data')
    parser.add_argument('--train_grads', type=str, required=True,
                        help='Path to training gradients (all_orig.pt)')
    parser.add_argument('--valid_grads', type=str, required=True,
                        help='Path to validation gradients (all_orig.pt)')
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

    # Load gradients
    print(f"Loading training gradients from {args.train_grads}...")
    train_grads = torch.load(args.train_grads)
    print(f"  Shape: {train_grads.shape}")

    print(f"Loading validation gradients from {args.valid_grads}...")
    valid_grads = torch.load(args.valid_grads)
    print(f"  Shape: {valid_grads.shape}")

    # Find valid (non-NaN, non-zero) training samples
    print("\nFiltering invalid gradients...")
    train_norms = torch.norm(train_grads, dim=1)
    train_valid_mask = ~torch.isnan(train_norms) & (train_norms > 0)

    n_total = len(train_grads)
    n_valid = train_valid_mask.sum().item()
    n_nan = torch.isnan(train_norms).sum().item()
    n_zero = (train_norms == 0).sum().item()

    print(f"  Total training samples: {n_total}")
    print(f"  Valid samples: {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  NaN samples: {n_nan} ({100*n_nan/n_total:.1f}%)")
    print(f"  Zero samples: {n_zero} ({100*n_zero/n_total:.1f}%)")

    # Move validation gradients to device
    valid_grads = valid_grads.to(device).float()

    # Initialize aggregated scores for ALL training samples (invalid ones get -inf)
    aggregated_scores = torch.full((n_total,), float('-inf'), dtype=torch.float32)

    # Compute influence scores in batches (only for valid samples)
    print(f"\nComputing influence scores (batch_size={args.batch_size})...")
    valid_indices = train_valid_mask.nonzero().squeeze().tolist()
    if isinstance(valid_indices, int):
        valid_indices = [valid_indices]

    for i in tqdm(range(0, len(valid_indices), args.batch_size), desc="Computing"):
        batch_end = min(i + args.batch_size, len(valid_indices))
        batch_indices = valid_indices[i:batch_end]

        # Get batch of training gradients
        batch_train = train_grads[batch_indices].to(device).float()

        # Compute influence: [batch, dim] @ [dim, n_valid] = [batch, n_valid]
        batch_scores = torch.matmul(batch_train, valid_grads.T)

        # Aggregate across validation samples
        if args.aggregation == 'mean':
            batch_agg = batch_scores.mean(dim=1)
        elif args.aggregation == 'sum':
            batch_agg = batch_scores.sum(dim=1)
        elif args.aggregation == 'max':
            batch_agg = batch_scores.max(dim=1)[0]

        aggregated_scores[batch_indices] = batch_agg.cpu()

    # Sort by aggregated score (descending)
    print("\nSorting by influence score...")
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
    }, scores_file)
    print(f"Saved raw scores to {scores_file}")

    # Print some stats
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Total training samples: {n_total}")
    print(f"  Valid samples: {len(sorted_scores)}")
    print(f"  Output file: {output_file}")
    print(f"  Top 5 influential samples (by original index):")
    for i in range(min(5, len(sorted_indices))):
        print(f"    #{i+1}: idx={sorted_indices[i].item()}, score={sorted_scores[i].item():.4f}")


if __name__ == "__main__":
    main()
