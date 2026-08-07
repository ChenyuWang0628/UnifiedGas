"""UnifiedGas network definitions.

The model is a compact CNN backbone followed by a two-level fully connected
trunk, two linear classifier heads of different capacity, and a fixed logit
fusion.  Two backbone variants are provided:

``SimpleCNN2D``
    For Dataset A (UCI Gas Sensor Array Drift), whose 128-dimensional feature
    vector is reshaped to a 16 x 8 map (16 sensors x 8 features) and processed
    with 3 x 3 convolutions.

``SimpleCNN1D``
    For Dataset B (UCI Twin Gas Sensor Arrays), whose 8-dimensional
    steady-state vector is treated as a 1 x 8 map and processed with 1 x 3
    convolutions.

Both expose a 256-dimensional embedding, so every downstream component --
including all domain-adaptation baselines -- shares an identical backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SimpleCNN1D",
    "SimpleCNN2D",
    "SensorTransformerBackbone",
    "MultiLevelFC",
    "GasClassifier",
    "build_backbone",
]


class SimpleCNN2D(nn.Module):
    """3 x 3 convolutional backbone for the 16 x 8 Dataset-A feature map."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(x).view(x.size(0), -1)


class SimpleCNN1D(nn.Module):
    """1 x 3 convolutional backbone for the 8-dimensional Dataset-B vector."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        k, p = (1, 3), (0, 1)
        pool = dict(kernel_size=(1, 2), stride=(1, 2))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, k, padding=p), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, k, padding=p), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, k, padding=p), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, k, padding=p), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(**pool),
            nn.Conv2d(64, 128, k, padding=p), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, k, padding=p), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(**pool),
            nn.Conv2d(128, 256, k, padding=p), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(x).view(x.size(0), -1)


class SensorTransformerBackbone(nn.Module):
    """Transformer encoder over the 16 sensor tokens of Dataset A.

    The dimensions match the architecture reported for TMSCA: input size 128,
    model width 128, six encoder layers, four attention heads, and a 256-unit
    feed-forward block. Each sensor contributes one token containing its eight
    engineered response features; attention therefore operates within each
    sample and never mixes samples from a minibatch.

    This is a capacity-matched diagnostic backbone, not a TMSCA
    re-implementation: it intentionally excludes TMSCA's cross-domain prior
    attention, decoder, k-NN LMMD, and ALR-CCA components.
    """

    def __init__(self, input_dim: int = 8, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 6,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 num_sensors: int = 16):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_sensors, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(d_model, 256)
        self.feature_dim = 256
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        elif x.dim() == 2:
            x = x.view(x.size(0), self.num_sensors, self.input_dim)
        if x.dim() != 3 or x.shape[1:] != (self.num_sensors, self.input_dim):
            raise ValueError(
                "Transformer backbone expects (B, 1, 16, 8), (B, 16, 8), "
                f"or (B, 128); got {tuple(x.shape)}"
            )
        tokens = self.input_projection(x) + self.position_embedding
        encoded = self.encoder(tokens)
        return self.output_projection(encoded.mean(dim=1))


def build_backbone(dataset: str, in_channels: int = 1,
                   backbone: str = "cnn", dropout_rate: float = 0.1) -> nn.Module:
    """Return the requested backbone for a dataset tag ('A' or 'B')."""
    backbone = backbone.lower()
    dataset = dataset.upper()
    if backbone == "transformer":
        if dataset != "A":
            raise ValueError("the capacity-matched Transformer is defined for Dataset A only")
        return SensorTransformerBackbone(dropout=dropout_rate)
    if backbone != "cnn":
        raise ValueError(f"backbone must be 'cnn' or 'transformer', got {backbone!r}")
    if dataset == "A":
        return SimpleCNN2D(in_channels)
    if dataset == "B":
        return SimpleCNN1D(in_channels)
    raise ValueError(f"dataset must be 'A' or 'B', got {dataset!r}")


class MultiLevelFC(nn.Module):
    """Fully connected trunk that exposes an intermediate feature at each level."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout_rate: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(dropout_rate)
            ))
            prev = h

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class GasClassifier(nn.Module):
    """UnifiedGas classifier: backbone + multi-level trunk + fused dual heads.

    ``forward(x)`` returns ``(out1, out2, fused)``.  When ``labels`` are given
    and the center regularizer is enabled it additionally returns the
    dual-level center loss, i.e. ``((out1, out2, fused), center_loss)``.
    """

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.1,
                 use_center_loss: bool = True, dataset: str = "A",
                 fusion_w1: float = 0.6, backbone: str = "cnn"):
        super().__init__()
        self.num_classes = num_classes
        self.use_center_loss = use_center_loss
        self.dataset = dataset.upper()
        self.backbone_name = backbone.lower()
        self.fusion_w1 = fusion_w1
        self.fusion_w2 = 1.0 - fusion_w1

        self.cnn = build_backbone(
            self.dataset,
            backbone=self.backbone_name,
            dropout_rate=dropout_rate,
        )
        self.fc_dims = [128, 64]
        self.fc = MultiLevelFC(self.cnn.feature_dim, self.fc_dims, dropout_rate)
        self.classifier1 = nn.Linear(self.fc_dims[0], num_classes)
        self.classifier2 = nn.Linear(self.fc_dims[1], num_classes)
        if use_center_loss:
            self.centers1 = nn.Parameter(torch.randn(num_classes, self.fc_dims[0]) * 0.1)
            self.centers2 = nn.Parameter(torch.randn(num_classes, self.fc_dims[1]) * 0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        if self.dataset == "A":
            if x.dim() == 2:            # (B, 128)
                x = x.view(x.size(0), 1, 16, 8)
            elif x.dim() == 3:          # (B, 16, 8)
                x = x.unsqueeze(1)
        else:
            if x.dim() == 2:            # (B, 8)
                x = x.view(x.size(0), 1, 1, x.size(1))
            elif x.dim() == 3:
                x = x.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        x = self._reshape(x)
        fc_f = self.fc(self.cnn(x))
        out1 = self.classifier1(fc_f[0])
        out2 = self.classifier2(fc_f[1])
        fused = self.fusion_w1 * out1 + self.fusion_w2 * out2
        if labels is not None and self.use_center_loss:
            cl1 = F.mse_loss(fc_f[0], self.centers1[labels]) / 2
            cl2 = F.mse_loss(fc_f[1], self.centers2[labels]) / 2
            return (out1, out2, fused), (cl1 + cl2) / 2
        return out1, out2, fused

    def get_all_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return [cnn_feature, f1, f2] -- the three levels aligned by MK-MMD."""
        x = self._reshape(x)
        cnn_f = self.cnn(x)
        return [cnn_f] + self.fc(cnn_f)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Class predictions from the fused logits."""
        self.eval()
        _, _, fused = self.forward(x)
        return fused.argmax(dim=1)
