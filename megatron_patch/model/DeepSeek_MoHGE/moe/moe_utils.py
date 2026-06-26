import math
from typing import Optional

import torch


def switch_load_balancing_group_loss_func(
    probs: torch.Tensor,
    tokens_per_group: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    min_moe_ffn_hidden_size: int,
    moe_ffn_hidden_size_interval: int,
    moe_ffn_hidden_size: list,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_groups].
        tokens_per_group (torch.Tensor): Number of tokens assigned to each group.
                                          Shape in [num_groups]
        topk (int): The number of groups selected for each token.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        sequence_partition_group (optional): The parallel group over which the sequence is
                                             partitioned. If None, no partitioning is applied.
                                             Defaults to None.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_group` local since we don't need the gradient for
        # `tokens_per_group`, saving one allreduce operation for `aggregated_probs_per_group`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_group, group=sequence_partition_group)

    num_tokens = probs.shape[0] * num_sub_sequence
    num_groups = probs.shape[1]

    # The formula of aux_loss: aux_loss = sum((probs_per_group/num_tokens) *
    # (tokens_per_group/(num_tokens*topk))) * num_groups * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    ffn_hidden_size_list_tensor = 1.0 * (torch.Tensor(moe_ffn_hidden_size) - min_moe_ffn_hidden_size + moe_ffn_hidden_size_interval) /  moe_ffn_hidden_size_interval
    ffn_hidden_size_list_tensor = ffn_hidden_size_list_tensor.to(probs.device)
    aggregated_probs_per_group = probs.sum(dim=0) * ffn_hidden_size_list_tensor
    aux_loss = torch.sum(aggregated_probs_per_group * tokens_per_group) * (
        num_groups * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss


def switch_load_balancing_loss_func_for_topk_groups(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    group_topk: int,
    num_groups: int,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
                                          Shape in [num_experts]
        topk (int): The number of experts selected for each token.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        sequence_partition_group (optional): The parallel group over which the sequence is
                                             partitioned. If None, no partitioning is applied.
                                             Defaults to None.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)
    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1] * group_topk / num_groups

    # The formula of aux_loss: aux_loss = sum((probs_per_expert/num_tokens) *
    # (tokens_per_expert/(num_tokens*topk))) * num_experts * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss
