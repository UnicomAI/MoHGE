# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from functools import partial

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
# from megatron.training.global_vars import get_tensorboard_writer

from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)

from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

from .experts import MyGroupedMLP, MySequentialMLP, MyTEGroupedMLP
from .shared_experts import SharedExpertMLP
from ..mlp import MLPSubmodules
from .moe_utils import switch_load_balancing_group_loss_func, switch_load_balancing_loss_func_for_topk_groups

# import globstep_config
# globstep_config.Glob_Step=None
@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class GroupTopKRouter(TopKRouter):
    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            moe_group_router_topk
            moe_group_router_load_balancing_type
            moe_group_router_score_function
            moe_group_router_enable_expert_bias
            num_moe_expert_groups
        """
        super().__init__(config=config)
        self.topk = self.config.moe_group_router_topk
        self.routing_type = self.config.moe_group_router_load_balancing_type
        self.score_function = self.config.moe_group_router_score_function
        self.input_jitter = None
        
        # print(self.config.num_moe_expert_groups, self.config.hidden_size)
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_expert_groups, self.config.hidden_size), dtype=torch.float32)
        )
        
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

        self.enable_expert_group_bias = self.config.moe_group_router_enable_expert_bias
        if self.enable_expert_group_bias:
            self.register_buffer(
                'local_tokens_per_expert_group',
                torch.zeros(self.config.num_moe_expert_groups, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_group_bias', torch.zeros(config.num_moe_expert_groups, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert_group = None
            self.expert_group_bias = None

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_group = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
            moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_group_bias,
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                switch_load_balancing_group_loss_func,
                probs=scores,
                tokens_per_group=tokens_per_group,
                topk=self.topk,
            )
            probs = self.apply_load_group_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def apply_load_group_balancing_loss(
            self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        # print("####")
        # print(globstep_config.Glob_Step,globstep_config.Train_iter)
        moe_group_aux_loss_coeff = self.config.moe_group_aux_loss_coeff if globstep_config.Glob_Step > globstep_config.Train_iter // 2 else 0
        min_moe_ffn_hidden_size = self.config.min_moe_ffn_hidden_size
        moe_ffn_hidden_size_interval = self.config.moe_ffn_hidden_size_interval
        moe_ffn_hidden_size = self.config.moe_ffn_hidden_size
        
        # if globstep_config.Glob_Step%100==0:
        #     print(globstep_config.Glob_Step,globstep_config.Train_iter,moe_group_aux_loss_coeff)
        
        
        if moe_group_aux_loss_coeff == 0:
            return activation
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_group_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        elif parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = load_balancing_loss_func(
            moe_ffn_hidden_size_interval=moe_ffn_hidden_size_interval, min_moe_ffn_hidden_size=min_moe_ffn_hidden_size, moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_aux_loss_coeff=moe_group_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        # print("###################pre load_group_balancing_loss")
        save_to_aux_losses_tracker(
            "load_group_balancing_loss",
            aux_loss / moe_group_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        # print("###################after load_group_balancing_loss")
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_expert_groups)
        

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
            """
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
            - tokens_per_expert (torch.Tensor): A tensor of shape [num_experts] containing
              the number of local tokens assigned to each expert before dropping and padding.
            """
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_group_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert_group += routing_map.sum(dim=0)
                
        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        # print("mg_logits:", logits)

        scores, routing_map = self.routing(logits)

        return scores, routing_map


class TopKRouterForDiffenerentGroup(TopKRouter):
    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            moe_group_router_topk
            moe_group_router_load_balancing_type
            moe_group_router_score_function
            moe_group_router_enable_expert_bias
            num_moe_expert_groups
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        self.num_moe_expert_groups = self.config.num_moe_expert_groups
        self.group_topk = self.config.moe_group_router_topk
        self.num_moe_experts = self.config.num_moe_experts

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def aux_loss_load_balancing_for_each_group(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            moe_router_topk_limited_devices=self.config.moe_router_topk_limited_devices,
            moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            # Apply load balancing loss
            aux_loss_func = partial(
                switch_load_balancing_loss_func_for_topk_groups,
                probs=probs,
                tokens_per_expert=tokens_per_expert,
                group_topk=self.group_topk,
                num_groups=self.num_moe_expert_groups,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def routing(self, logits: torch.Tensor, group_scores: torch.Tensor, group_routing_map: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        num_moe_experts_in_each_group = self.config.num_moe_experts // self.num_moe_expert_groups
        logits = logits.view(-1, self.num_moe_expert_groups, num_moe_experts_in_each_group)
        logits = F.softmax(logits, -1)
        logits = logits * group_scores.unsqueeze(-1) * group_routing_map.unsqueeze(-1)
        logits = logits.view(-1, self.config.num_moe_experts)
        logits = logits / torch.sum(logits, dim=-1, keepdim=True)
        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)
        if self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing_for_each_group(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                moe_router_topk_scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        
        indices = torch.arange(self.num_moe_expert_groups).to(scores.device)
        rearranged_indices = torch.cat([indices[:self.num_moe_expert_groups//2].unsqueeze(-1), indices[self.num_moe_expert_groups//2:].flip(-1).unsqueeze(-1)], dim=-1).flatten()
        
        scores = scores.view(-1, self.num_moe_expert_groups, num_moe_experts_in_each_group)
        scores = scores.permute(0, 2, 1)
        scores = scores.gather(-1, rearranged_indices.expand_as(scores))
        scores = scores.view(-1, self.config.num_moe_experts)
        
        routing_map = routing_map.view(-1, self.num_moe_expert_groups, num_moe_experts_in_each_group)
        routing_map = routing_map.permute(0, 2, 1)
        routing_map = routing_map.gather(-1, rearranged_indices.expand_as(routing_map))
        routing_map = routing_map.view(-1, self.config.num_moe_experts)
        
        return scores, routing_map

    def forward(self, input: torch.Tensor, group_scores, group_routing_map):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """

        # Apply input jitter

        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map = self.routing(logits, group_scores, group_routing_map)
        return scores, routing_map


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)
        self.group_router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
            self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.group_router = GroupTopKRouter(config=self.config)

        self.router = TopKRouterForDiffenerentGroup(config=self.config)

        self.n_experts_in_group = self.config.num_moe_experts // self.config.num_moe_expert_groups

        # Initialize experts
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules.experts, MLPSubmodules):
                self.experts = MyTEGroupedMLP(self.local_expert_indices, self.num_local_experts, self.config,
                                              self.submodules.experts)
            else:
                self.experts = MyGroupedMLP(self.self.local_expert_indices, self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules.experts, MLPSubmodules)
            self.experts = MySequentialMLP(
                self.local_expert_indices, self.num_local_experts, self.config, self.submodules.experts
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if (
                self.training
                and self.config.tensor_model_parallel_size > 1
                and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        # process MoE
        def custom_forward(hidden_states):
            group_probs, group_routing_map = self.group_router(hidden_states)
            probs, routing_map = self.router(hidden_states, group_probs, group_routing_map)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            # print("token_permutation:",dispatched_input.shape,tokens_per_expert.shape)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)
        return output, mlp_bias


