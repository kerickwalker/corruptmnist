from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)
        self.images = None
        self.targets = None
        self._load_data()

    def _load_data(self) -> None:
        """Load all train and test data files."""
        # Load training data
        train_images_list = []
        train_targets_list = []
        
        for i in range(6):  # train_images_0.pt to train_images_5.pt
            train_images_list.append(torch.load(self.data_path / f"train_images_{i}.pt"))
            train_targets_list.append(torch.load(self.data_path / f"train_target_{i}.pt"))
        
        # Load test data
        test_images = torch.load(self.data_path / "test_images.pt")
        test_targets = torch.load(self.data_path / "test_target.pt")
        
        # Concatenate all data
        self.images = torch.cat(train_images_list + [test_images], dim=0)
        self.targets = torch.cat(train_targets_list + [test_targets], dim=0)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index], self.targets[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Normalize images: mean 0, std 1
        mean = self.images.float().mean()
        std = self.images.float().std()
        
        normalized_images = (self.images.float() - mean) / std
        
        # Save normalized data
        torch.save(normalized_images, output_folder / "images.pt")
        torch.save(self.targets, output_folder / "targets.pt")
        
        print(f"Preprocessing complete. Data saved to {output_folder}")
        print(f"Images shape: {normalized_images.shape}")
        print(f"Targets shape: {self.targets.shape}")
        print(f"Images mean: {normalized_images.mean():.6f}, std: {normalized_images.std():.6f}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Main preprocessing function."""
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
