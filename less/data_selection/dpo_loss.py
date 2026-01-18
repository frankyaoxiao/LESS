"""
DPO (Direct Preference Optimization) loss computation for gradient collection.

The DPO loss is:
    L_DPO = -log(sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))))

Where:
    - pi_chosen: log prob of chosen response under policy model
    - ref_chosen: log prob of chosen response under reference model
    - pi_rejected: log prob of rejected response under policy model
    - ref_rejected: log prob of rejected response under reference model
    - beta: temperature parameter (controls KL penalty strength)

For LESS with LoRA:
    - Reference model = base model with LoRA disabled
    - Policy model = base model with LoRA enabled
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel


def get_batch_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    average_log_prob: bool = True,
) -> torch.Tensor:
    """
    Compute the log probabilities for the response tokens.

    Args:
        model: Language model
        input_ids: Token IDs [batch_size, seq_len]
        labels: Labels with -100 for masked (prompt) tokens [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        average_log_prob: If True, return per-token average; if False, return sum

    Returns:
        Tensor of shape [batch_size] containing log probs for each sequence
    """
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # Shift for causal LM (predict next token)
    shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]

    # Gather log probs for the actual tokens
    # We need to handle the case where labels are -100 (masked tokens)
    # Create a version of labels where -100 is replaced with 0 for gathering
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_len-1]

    # Mask out padding/prompt tokens (where labels == -100)
    mask = (shift_labels != -100).float()

    # Sum log probs over sequence (only for non-masked tokens)
    sum_log_probs = (token_log_probs * mask).sum(dim=-1)  # [batch_size]

    if average_log_prob:
        # Compute per-token average (more numerically stable for DPO)
        num_tokens = mask.sum(dim=-1).clamp(min=1)  # Avoid division by zero
        return sum_log_probs / num_tokens
    else:
        return sum_log_probs


def compute_dpo_loss(
    model: PeftModel,
    batch: Dict[str, torch.Tensor],
    beta: float = 5.0,
    reference_free: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute DPO loss for a batch of preference pairs.

    The DPO loss is:
        L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    where:
        log_ratio_chosen = pi(chosen) - ref(chosen)
        log_ratio_rejected = pi(rejected) - ref(rejected)

    Args:
        model: PeftModel with LoRA adapter
        batch: Dictionary containing:
            - chosen_input_ids: [batch_size, seq_len]
            - chosen_labels: [batch_size, seq_len]
            - chosen_attention_mask: [batch_size, seq_len]
            - rejected_input_ids: [batch_size, seq_len]
            - rejected_labels: [batch_size, seq_len]
            - rejected_attention_mask: [batch_size, seq_len]
        beta: DPO temperature parameter (default 5.0)
        reference_free: If True, skip reference model computation

    Returns:
        loss: Scalar loss tensor with gradient tracking
        metrics: Dictionary with additional metrics for logging
    """
    # Extract batch components
    chosen_input_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_input_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    # Step 1: Get reference log probs (LoRA disabled)
    if not reference_free:
        with torch.no_grad():
            # Disable LoRA adapter to get reference model behavior
            model.disable_adapter_layers()

            ref_chosen_logps = get_batch_log_probs(
                model, chosen_input_ids, chosen_labels, chosen_attention_mask
            )
            ref_rejected_logps = get_batch_log_probs(
                model, rejected_input_ids, rejected_labels, rejected_attention_mask
            )

            # Re-enable LoRA adapter
            model.enable_adapter_layers()
    else:
        # Reference-free DPO: set reference log probs to 0
        ref_chosen_logps = torch.zeros(chosen_input_ids.shape[0], device=chosen_input_ids.device)
        ref_rejected_logps = torch.zeros(rejected_input_ids.shape[0], device=rejected_input_ids.device)

    # Step 2: Get policy log probs (LoRA enabled)
    policy_chosen_logps = get_batch_log_probs(
        model, chosen_input_ids, chosen_labels, chosen_attention_mask
    )
    policy_rejected_logps = get_batch_log_probs(
        model, rejected_input_ids, rejected_labels, rejected_attention_mask
    )

    # Step 3: Compute DPO loss
    # Log ratios
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps

    # DPO logits (reward difference)
    logits = chosen_log_ratios - rejected_log_ratios

    # DPO loss: -log(sigmoid(beta * logits))
    losses = -F.logsigmoid(beta * logits)
    loss = losses.mean()

    # Compute metrics for logging
    with torch.no_grad():
        chosen_rewards = beta * chosen_log_ratios
        rejected_rewards = beta * rejected_log_ratios
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards

    metrics = {
        "loss": loss.detach(),
        "chosen_rewards": chosen_rewards.mean().detach(),
        "rejected_rewards": rejected_rewards.mean().detach(),
        "reward_accuracies": reward_accuracies.mean().detach(),
        "reward_margins": reward_margins.mean().detach(),
        "policy_chosen_logps": policy_chosen_logps.mean().detach(),
        "policy_rejected_logps": policy_rejected_logps.mean().detach(),
    }

    return loss, metrics


def compute_dpo_loss_simple(
    model: PeftModel,
    batch: Dict[str, torch.Tensor],
    beta: float = 5.0,
) -> torch.Tensor:
    """
    Simplified DPO loss computation (returns only loss tensor).

    This is a convenience function for gradient collection where we only need the loss.

    Args:
        model: PeftModel with LoRA adapter
        batch: Dictionary with preference pair tensors
        beta: DPO temperature parameter

    Returns:
        Scalar loss tensor with gradient tracking
    """
    loss, _ = compute_dpo_loss(model, batch, beta=beta)
    return loss
