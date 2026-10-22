"""
Bitemporal Image Transformer (BIT-CD) Architecture for Remote Sensing Change Detection.
Reference:
    Hao Chen and Zhenwei Shi.
    "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection."
    Remote Sensing, 2020 / IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2021.
    Official Repository: https://github.com/justchenhao/BIT_CD (MIT License)

Matches the official BASE_Transformer variant: base_transformer_pos_s4_dd8_dedim8
Implemented in pure PyTorch (no external einops dependency required).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class TwoLayerConv2d(nn.Sequential):
    """Two-layer convolution block with BatchNorm and ReLU for prediction head."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                stride=1, 
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                stride=1
            )
        )


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), x2, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Cross_Attention(nn.Module):
    """Multi-head cross attention between pixel features and semantic tokens."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 8, dropout: float = 0.0, softmax: bool = True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.softmax = softmax

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        d = q.shape[-1] // h
        q = q.view(b, n, h, d).permute(0, 2, 1, 3)
        k = k.view(b, m.shape[1], h, d).permute(0, 2, 1, 3)
        v = v.view(b, m.shape[1], h, d).permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1) if self.softmax else dots
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        return self.to_out(out)


class Attention(nn.Module):
    """Multi-head self attention over concatenated semantic tokens."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        d = qkv[0].shape[-1] // h
        q, k, v = [t.view(b, n, h, d).permute(0, 2, 1, 3) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer encoder for context modeling over semantic tokens."""
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder projecting semantic tokens back to pixel space."""
    def __init__(
        self, 
        dim: int, 
        depth: int, 
        heads: int, 
        dim_head: int, 
        mlp_dim: int, 
        dropout: float, 
        softmax: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(
                    dim, 
                    Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, softmax=softmax)
                )),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x: torch.Tensor, m: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
        return x


class BitemporalImageTransformer(nn.Module):
    """
    Official Bitemporal Image Transformer (BIT-CD) Network.
    Matches architecture configuration: base_transformer_pos_s4_dd8_dedim8
    Strictly compatible with official BIT-CD checkpoints from justchenhao/BIT_CD.
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 2, 
        token_len: int = 4, 
        token_dim: int = 32,
        num_decoder_layers: int = 8,
        decoder_dim_head: int = 8,
        enc_depth: int = 1,
        resnet_stages_num: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.token_len = token_len
        self.token_dim = token_dim
        self.resnet_stages_num = resnet_stages_num

        # Siamese ResNet-18 feature extractor
        self.resnet = models.resnet18(weights=None)

        # Feature projection to token dimension (resnet_stages_num=4 outputs 256 channels)
        backbone_channels = 256 if resnet_stages_num == 4 else 512
        self.conv_pred = nn.Conv2d(backbone_channels, token_dim, kernel_size=3, padding=1)

        # Spatial attention tokenizer
        self.conv_a = nn.Conv2d(token_dim, token_len, kernel_size=1, padding=0, bias=False)

        # Learned positional embedding for concatenated tokens (T1 + T2 = token_len * 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len * 2, token_dim))

        # Transformer encoder (1 layer, 8 heads, dim_head=64, mlp_dim=64)
        self.transformer = Transformer(
            dim=token_dim,
            depth=enc_depth,
            heads=8,
            dim_head=64,
            mlp_dim=token_dim * 2,
            dropout=0.0
        )

        # Transformer decoder (8 layers, 8 heads, decoder_dim_head=8, mlp_dim=64)
        self.transformer_decoder = TransformerDecoder(
            dim=token_dim,
            depth=num_decoder_layers,
            heads=8,
            dim_head=decoder_dim_head,
            mlp_dim=token_dim * 2,
            dropout=0.0,
            softmax=True
        )

        # Classifier head
        self.classifier = TwoLayerConv2d(in_channels=token_dim, out_channels=num_classes)

        # Spatial upsampling layers
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Siamese backbone feature extraction for a single temporal scene."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)
        x_8 = self.resnet.layer2(x_4)
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)

        x = self.upsamplex2(x_8)
        x = self.conv_pred(x)
        return x

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bi-temporal input pair:
        Args:
            t1: [B, 3, H, W] Earlier scene
            t2: [B, 3, H, W] Later scene
        Returns:
            logits: [B, 2, H, W] Change logits (channel 0: no-change, channel 1: change)
        """
        input_h, input_w = t1.shape[2], t1.shape[3]

        # 1. Siamese feature extraction
        f1 = self.forward_single(t1)  # [B, token_dim, H_feat, W_feat]
        f2 = self.forward_single(t2)  # [B, token_dim, H_feat, W_feat]
        b, c, h, w = f1.shape

        # 2. Tokenizer: compute spatial attention and pool semantic tokens
        attn1 = torch.softmax(self.conv_a(f1).view(b, self.token_len, -1), dim=-1)
        token1 = torch.bmm(attn1, f1.view(b, c, -1).transpose(1, 2))  # [B, token_len, token_dim]

        attn2 = torch.softmax(self.conv_a(f2).view(b, self.token_len, -1), dim=-1)
        token2 = torch.bmm(attn2, f2.view(b, c, -1).transpose(1, 2))  # [B, token_len, token_dim]

        # 3. Transformer encoder: temporal interaction over concatenated tokens
        tokens = torch.cat([token1, token2], dim=1) + self.pos_embedding
        tokens = self.transformer(tokens)
        token1, token2 = tokens.chunk(2, dim=1)

        # 4. Transformer decoder: project semantic tokens back to spatial feature maps
        f1_tokens = f1.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        f2_tokens = f2.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        f1_dec = self.transformer_decoder(f1_tokens, token1).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        f2_dec = self.transformer_decoder(f2_tokens, token2).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # 5. Differencing and classification
        diff = torch.abs(f1_dec - f2_dec)
        diff = self.upsamplex4(diff)
        logits = self.classifier(diff)

        # Ensure spatial dimensions match input exactly
        if logits.shape[2:] != (input_h, input_w):
            logits = F.interpolate(logits, size=(input_h, input_w), mode='bilinear', align_corners=False)

        return logits


# Aliases for official naming compatibility
BASE_Transformer = BitemporalImageTransformer
BIT = BitemporalImageTransformer
