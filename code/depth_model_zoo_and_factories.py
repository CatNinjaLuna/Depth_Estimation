"""
depth_model_zoo_and_factories.py

Model zoo for depth estimation algorithms used in the benchmarking
framework. The goal of this module is to provide a set of representative
architectures mirroring the categories summarized in the manuscript:

    - CNN Baseline
    - Lightweight CNN
    - Transformer-based model
    - Hybrid CNN-Transformer

The implementations are intentionally compact yet expressive, emphasizing
clear structure over maximal performance. All architectures expose a
uniform forward interface:

    forward(rgb: Tensor[B, 3, H, W]) -> Tensor[B, 1, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Common building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution used in lightweight architectures.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PositionalEncoding2d(nn.Module):
    """
    Simple 2D sine-cosine positional encoding for transformer-based models.
    """

    def __init__(self, num_feats: int = 64) -> None:
        super().__init__()
        self.num_feats = num_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        device = x.device

        y_embed = torch.linspace(0, 1, steps=h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.linspace(0, 1, steps=w, device=device).unsqueeze(0).repeat(h, 1)

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (torch.sin(pos_x[..., 0::2]), torch.cos(pos_x[..., 1::2])),
            dim=-1,
        ).flatten(-2)
        pos_y = torch.stack(
            (torch.sin(pos_y[..., 0::2]), torch.cos(pos_y[..., 1::2])),
            dim=-1,
        ).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, 2*num_feats)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, pos], dim=1)  # concatenate along channel dim


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block adapted to 2D feature maps.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Depth estimation architectures
# ---------------------------------------------------------------------------


class CNNBaselineDepthNet(nn.Module):
    """
    UNet-style baseline CNN architecture.
    """

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(3, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 2),
        )
        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvBlock(base_channels * 4, base_channels * 4),
        )
        self.enc4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, stride=2),
            ConvBlock(base_channels * 8, base_channels * 8),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4),
        )
        self.dec2 = nn.Sequential(
            ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2),
        )
        self.dec1 = nn.Sequential(
            ConvBlock(base_channels * 2 + base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = F.interpolate(e4, scale_factor=2.0, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2.0, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2.0, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        depth = self.out_conv(d1)
        depth = F.relu(depth)  # enforce non-negative depth
        return depth


class LightweightDepthNet(nn.Module):
    """
    Lightweight depth network built predominantly from depthwise separable
    convolutions. This model aims to emulate "FastDepth"/mobile-style
    architectures discussed in the literature.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock(3, base_channels, kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, stride=2),
            DepthwiseSeparableConv(base_channels * 2, base_channels * 2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, stride=2),
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4, stride=1),
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4, stride=1),
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        y = self.up2(x3)
        y = self.up1(y)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        depth = self.out_conv(y)
        depth = F.relu(depth)
        return depth


class TransformerDepthNet(nn.Module):
    """
    Transformer-based depth estimation model.

    This implementation follows a simple encoder-transformer-decoder scheme:
        - CNN backbone to extract low-resolution features
        - Transformer encoder operating on flattened tokens
        - CNN decoder to upsample back to full resolution
    """

    def __init__(
        self,
        base_channels: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels, stride=2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)

        self.pos_enc = PositionalEncoding2d(num_feats=base_channels * 2)
        token_dim = base_channels * 4 + 2 * base_channels * 2
        self.proj = nn.Conv2d(token_dim, base_channels * 4, kernel_size=1)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(dim=base_channels * 4, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)  # (B, C, H/8, W/8)

        x_pos = self.pos_enc(x3)
        x_proj = self.proj(x_pos)

        b, c_proj, h_low, w_low = x_proj.shape
        tokens = x_proj.flatten(2).transpose(1, 2)  # (B, N, C)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        feat = tokens.transpose(1, 2).view(b, c_proj, h_low, w_low)

        y = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = self.dec2(y)
        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = self.dec1(y)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        depth = self.out_conv(y)
        depth = F.relu(depth)
        return depth


class HybridCNNTransformerDepthNet(nn.Module):
    """
    Hybrid CNN-Transformer architecture combining local CNN features with
    global self-attention over a reduced-resolution representation.
    """

    def __init__(
        self,
        base_channels: int = 64,
        transformer_depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.local_stem = nn.Sequential(
            ConvBlock(3, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.local_down = ConvBlock(base_channels, base_channels * 2, stride=2)

        self.global_enc = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pos_enc = PositionalEncoding2d(num_feats=base_channels * 2)
        token_dim = base_channels * 4 + 2 * base_channels * 2
        self.proj = nn.Conv2d(token_dim, base_channels * 4, kernel_size=1)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=base_channels * 4, num_heads=num_heads, mlp_ratio=4.0
                )
                for _ in range(transformer_depth)
            ]
        )

        self.decoder = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=4),
            ConvBlock(base_channels, base_channels),
        )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        local_feat = self.local_stem(x)
        local_down = self.local_down(local_feat)
        global_feat = self.global_enc(local_down)

        global_pos = self.pos_enc(global_feat)
        global_proj = self.proj(global_pos)

        b, c_g, h_g, w_g = global_proj.shape
        tokens = global_proj.flatten(2).transpose(1, 2)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = tokens.transpose(1, 2).view(b, c_g, h_g, w_g)

        y = self.decoder(tokens)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        depth = self.out_conv(y)
        depth = F.relu(depth)
        return depth


# ---------------------------------------------------------------------------
# Model registry and factory
# ---------------------------------------------------------------------------


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn_baseline": CNNBaselineDepthNet,
    "lightweight_cnn": LightweightDepthNet,
    "transformer": TransformerDepthNet,
    "hybrid_cnn_transformer": HybridCNNTransformerDepthNet,
}


@dataclass
class DepthModelConfig:
    arch: str
    base_channels: int = 64
    transformer_depth: int = 4
    num_heads: int = 4


def build_depth_model(config: DepthModelConfig) -> nn.Module:
    """
    Instantiate a depth model from the registry using a high-level config.
    """
    arch = config.arch.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )

    if arch == "cnn_baseline":
        return CNNBaselineDepthNet(base_channels=config.base_channels)
    if arch == "lightweight_cnn":
        return LightweightDepthNet(base_channels=config.base_channels)
    if arch == "transformer":
        return TransformerDepthNet(
            base_channels=config.base_channels,
            depth=config.transformer_depth,
            num_heads=config.num_heads,
        )
    if arch == "hybrid_cnn_transformer":
        return HybridCNNTransformerDepthNet(
            base_channels=config.base_channels,
            transformer_depth=config.transformer_depth,
            num_heads=config.num_heads,
        )
    # Should never reach here
    raise RuntimeError(f"Unhandled architecture: {arch}")
