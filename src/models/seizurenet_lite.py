"""The parameter-constrained sequential seizure-window classifier.

The architecture follows the locked specification in ``README.md``.  In
particular, all convolutions are valid (no zero padding), the first kernel
collapses the four EEG channels, and the final layer emits a raw logit.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SeizureNetLite(nn.Module):
    """A 2,991-parameter sequential convolutional EEG classifier."""

    def __init__(self, output_bias: float | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 17), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(16, 10, kernel_size=(1, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(10, 10, kernel_size=(1, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(10)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(10, 10, kernel_size=(1, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(10)
        self.dropout4 = nn.Dropout(p=0.2)

        self.output = nn.Conv2d(10, 1, kernel_size=(1, 1), bias=True)
        self.reset_parameters(output_bias=output_bias)

    def reset_parameters(self, output_bias: float | None = None) -> None:
        """Use stable He initialization and an optional sampled-prior bias."""
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        for batch_norm in (self.bn1, self.bn2, self.bn3, self.bn4):
            nn.init.ones_(batch_norm.weight)
            nn.init.zeros_(batch_norm.bias)
        nn.init.kaiming_uniform_(self.output.weight, a=math.sqrt(5))
        if output_bias is None:
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.constant_(self.output.bias, float(output_bias))

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters in the complete network."""
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 4, 256):
            raise ValueError(
                "SeizureNetLite expects input shape [B, 1, 4, 256], "
                f"got {tuple(x.shape)}."
            )

        x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(torch.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(torch.relu(self.bn4(self.conv4(x))))
        logits = self.output(x)
        if logits.shape[1:] != (1, 1, 1):
            raise RuntimeError(
                "The locked valid-convolution geometry should end at [B, 1, 1, 1], "
                f"got {tuple(logits.shape)}."
            )
        return logits.reshape(-1)
