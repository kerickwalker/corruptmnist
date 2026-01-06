from pathlib import Path
from typing import Optional

import typer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from corruptmnist_classifier.model import Model
from corruptmnist_classifier.data import MyDataset
from corruptmnist_classifier.model import create_model


def train(
    data_path: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/figures"),
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> None:
    """Train a small classifier on the processed dataset and save artifacts.

    The function expects `images.pt` and `targets.pt` to exist under `data_path`.
    It saves the model state dict to `models_dir/model.pt` and a loss plot
    to `figures_dir/training_loss.png`.
    """
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    figures_dir = Path(figures_dir)

    images_fp = data_path / "images.pt"
    targets_fp = data_path / "targets.pt"

    if not images_fp.exists() or not targets_fp.exists():
        raise FileNotFoundError(f"Processed data not found in {data_path}")

    images = torch.load(images_fp)
    targets = torch.load(targets_fp)

    # Ensure image shape is (N, C, H, W)
    if images.ndim == 3:  # (N, H, W)
        images = images.unsqueeze(1)

    images = images.float()
    targets = targets.long()

    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=10, in_channels=images.shape[1])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}")

    # Ensure output directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_fp = models_dir / "model.pt"
    torch.save(model.state_dict(), model_fp)
    print(f"Saved trained model to {model_fp}")

    # Plot training loss
    fig_fp = figures_dir / "training_loss.png"
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_fp)
    plt.close()
    print(f"Saved training loss plot to {fig_fp}")


if __name__ == "__main__":
    typer.run(train)

def train():
    MyDataset("data/raw")
    Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
