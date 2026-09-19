import torch
import torch.nn as nn
from torchvision import models


# ================================================================
# CHANNEL ATTENTION
# ================================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


# ================================================================
# SPATIAL ATTENTION
# ================================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# ================================================================
# MULTI-HEAD SELF-ATTENTION
# ================================================================

class MHSA(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, C, H, W = x.shape

        sequence = x.flatten(2).transpose(1, 2)

        normalized = self.norm(sequence)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized
        )

        sequence = (
            sequence +
            self.dropout(attention_output)
        )

        x = (
            sequence
            .transpose(1, 2)
            .reshape(B, C, H, W)
        )

        return x


# ================================================================
# MODEL 3 — EFFICIENTNET-B3 + HYBRID ATTENTION
# ================================================================

class EfficientNet_HybridAttention(nn.Module):

    def __init__(
        self,
        num_classes=4,
        num_heads=8
    ):
        super().__init__()

        # No internet download required here.
        # The checkpoint will provide all learned weights.
        backbone = models.efficientnet_b3(
            weights=None
        )

        self.features = backbone.features

        feature_channels = 1536

        self.channel_attention = ChannelAttention(
            feature_channels
        )

        self.spatial_attention = SpatialAttention()

        self.mhsa = MHSA(
            feature_channels,
            num_heads=num_heads,
            dropout=0.1
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(
                feature_channels,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        x = self.mhsa(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x