from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )


class ChessCNN(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[ChessCNN, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ChessCNN(num_classes=int(checkpoint.get("num_classes", 14)), dropout=float(checkpoint.get("dropout", 0.30)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
