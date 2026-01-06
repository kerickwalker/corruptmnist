from pathlib import Path
from typing import Optional

import typer
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from corruptmnist_classifier.model import create_model


def visualize(
	model_path: Path = Path("models/model.pt"),
	data_path: Path = Path("data/processed"),
	out_path: Path = Path("reports/figures/tsne.png"),
	device: Optional[str] = None,
	max_samples: int = 2000,
	perplexity: int = 30,
	random_state: int = 42,
) -> None:
	"""Load a trained model, extract features, run t-SNE and save a 2D plot.

	The script expects `images.pt` and `targets.pt` in `data_path` and a
	model state dict at `model_path` (as produced by `train.py`).
	"""
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model_path = Path(model_path)
	data_path = Path(data_path)
	out_path = Path(out_path)

	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	images_fp = data_path / "images.pt"
	targets_fp = data_path / "targets.pt"
	if not images_fp.exists() or not targets_fp.exists():
		raise FileNotFoundError(f"Processed data not found in {data_path}")

	images = torch.load(images_fp)
	targets = torch.load(targets_fp)

	if images.ndim == 3:
		images = images.unsqueeze(1)

	images = images.float()
	targets = targets.long()

	# limit samples for t-SNE runtime
	n_samples = min(len(images), max_samples)
	images = images[:n_samples]
	targets = targets[:n_samples]

	dataset = TensorDataset(images, targets)
	loader = DataLoader(dataset, batch_size=256, shuffle=False)

	# build model and load weights
	model = create_model(in_channels=images.shape[1])
	state = torch.load(model_path, map_location="cpu")
	model.load_state_dict(state)
	model.to(device)
	model.eval()

	feats = []
	labels = []
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			# forward up to penultimate features
			x_feat = model.features(xb)
			x_flat = torch.flatten(x_feat, 1)
			# apply the first linear -> hidden (classifier[1]) and activation (classifier[2])
			hidden = model.classifier[1](x_flat)
			try:
				hidden = model.classifier[2](hidden)
			except Exception:
				pass
			feats.append(hidden.cpu())
			labels.append(yb)

	feats = torch.cat(feats, dim=0).numpy()
	labels = torch.cat(labels, dim=0).numpy()

	# t-SNE dimensionality reduction
	try:
		from sklearn.manifold import TSNE
	except Exception as exc:  # pragma: no cover - informative error
		raise RuntimeError("sklearn is required for t-SNE visualization") from exc

	tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
	z = tsne.fit_transform(feats)

	# plot
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 8))
	scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
	plt.title("t-SNE of penultimate features")
	plt.xticks([])
	plt.yticks([])
	cbar = plt.colorbar(scatter, ticks=range(int(labels.max()) + 1))
	cbar.set_label("label")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()
	print(f"Saved t-SNE visualization to {out_path}")


if __name__ == "__main__":
	typer.run(visualize)

