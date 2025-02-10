from dataclasses import dataclass

@dataclass(frozen=True)
class UNetRTrainerConfig:
    image_size:int
    patch_size:int
    hidden_dim:int
    num_channels:int
    num_layers:int
    num_heads:int
    mlp_dim:int
    dropout_rate:float