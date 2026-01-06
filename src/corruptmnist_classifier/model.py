from __future__ import annotations


import torch
from torch import nn


class Model(nn.Module):
    """A small CNN image classifier for (corrupted) MNIST-style inputs.

    Expects input tensors of shape (B, C, H, W) where C is typically 1.
    Returns raw logits of shape (B, num_classes).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, hidden_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor, shape (B, C, H, W)

        Returns:
            logits tensor, shape (B, num_classes)
        """
        return self.classifier(self.features(x))


def create_model(num_classes: int = 10, in_channels: int = 1, hidden_dim: int = 128) -> nn.Module:
    """Factory helper to create the model."""
    return Model(num_classes=num_classes, in_channels=in_channels, hidden_dim=hidden_dim)


if __name__ == "__main__":
    model = create_model()

    # Print model architecture
    print(model)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")

    # quick sanity forward pass and shape
    x = torch.rand(1, 1, 28, 28)
    out = model(x)
    print(f"Output shape of model: {out.shape}")
