from dataclasses import dataclass
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend


@dataclass
class MyMoETransformerConfig(MLATransformerConfig):
    
    moe_ffn_hidden_size: list = None

    moe_layer_freq: int = None

    moe_group_router_topk: int = None

    moe_group_router_load_balancing_type: str = None

    moe_group_router_score_function: str = None

    moe_group_router_enable_expert_bias: bool = False

    num_moe_expert_groups: int = None
    
    moe_group_aux_loss_coeff: float = None
    
    min_moe_ffn_hidden_size: int = None
    
    moe_ffn_hidden_size_interval: int = None
    
    
