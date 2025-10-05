"""
This file is an Autoencoder (MLP) for anomaly detection / feature compression.

- Architecture: Encoder → bottleneck (latent_dim) → mirrored decoder.
- Encoder blocks: Linear → (BatchNorm/LayerNorm/none) → LeakyReLU → (Dropout).
  Decoder uses Linear → LeakyReLU, with a final Linear to reconstruct inputs.
- Design intent:
    * Train on benign data so "normal" patterns reconstruct well; anomalies reconstruct poorly.
    * Dropout only in the encoder to regularize the learned representation.
    * Optional normalization ("batch" | "layer" | "none") in encoder for stabler training.
    * Xavier init for Linear layers; norms start at identity-like settings.

Args:
    input_dim: number of input features.
    hidden_dims: tuple of encoder layer sizes (decoder mirrors this order).
    latent_dim: size of the bottleneck (compression level).
    dropout: dropout probability applied in encoder blocks.
    norm: normalization type used in encoder blocks ("batch", "layer", "none").
"""

import torch
import torch.nn as nn
from typing import Tuple, Literal

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        latent_dim: int = 16,
        dropout: float = 0.1,
        norm: Literal["batch", "layer", "none"] = "batch",
    ):
        super().__init__()
        assert len(hidden_dims) >= 2

        # --- Encoder -----------------------------------------------------------
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            if norm == "batch":
                enc_layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.LeakyReLU(0.1, inplace=True))
            if dropout and dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        # Bottleneck
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # --- Decoder (mirror) --------------------------------------------------
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LeakyReLU(0.1, inplace=True)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier init for linears; zeros for norms/bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
