"""Temporal Convolutional Network for LOB mid-price direction prediction.

See LOB_TCN_DESIGN.md Section 4.2.

Causality is enforced by left-padding every dilated 1-D conv and trimming the
right edge of the convolution output. With L blocks of kernel k and dilation
[1, 2, 4, ..., 2^(L-1)] the receptive field is 1 + 2 * (k - 1) * (2^L - 1)
(two convs per block).
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """1-D convolution with left padding so output[t] depends only on input[<=t]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two dilated causal convs with batch norm, residual, dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        return F.relu(out + identity)


class TCNModel(nn.Module):
    """Stack of dilated TCN blocks with one classification head per horizon."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        n_levels: int = 4,
        n_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        horizons: Iterable[int] = (10, 50, 100),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout
        self.horizons = tuple(horizons)

        blocks = []
        in_ch = n_features
        for i in range(n_levels):
            dilation = 2**i
            blocks.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = n_channels
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.heads = nn.ModuleDict()
        for h in self.horizons:
            head = nn.Sequential(
                nn.Linear(n_channels, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )
            self.heads[f"horizon_{h}"] = head

        self._init_heads()

    def _init_heads(self) -> None:
        for module in self.heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, F) -> (B, F, T) for conv1d
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, F) got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        pooled = self.global_pool(x).squeeze(-1)  # (B, C)
        return {name: head(pooled) for name, head in self.heads.items()}

    def receptive_field(self) -> int:
        # 2 convs per block, each adds (k-1)*dilation; sum across levels
        rf = 1
        for i in range(self.n_levels):
            d = 2**i
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
