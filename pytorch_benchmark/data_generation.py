"""
Data generation module for structured (tabular) and unstructured (image) data.

All generation is deterministic given a fixed seed to support reproducibility testing.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification

from pytorch_benchmark.config import (
    RANDOM_SEED,
    STRUCTURED_NUM_SAMPLES,
    STRUCTURED_NUM_FEATURES,
    STRUCTURED_NUM_CLASSES,
    UNSTRUCTURED_NUM_SAMPLES,
    UNSTRUCTURED_IMAGE_SIZE,
    UNSTRUCTURED_NUM_CLASSES,
    BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Seeding utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int = RANDOM_SEED):
    """Set all random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Structured (Tabular) Data
# ---------------------------------------------------------------------------

def generate_structured_data(
    n_samples: int = STRUCTURED_NUM_SAMPLES,
    n_features: int = STRUCTURED_NUM_FEATURES,
    n_classes: int = STRUCTURED_NUM_CLASSES,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Generate a synthetic multi-class classification dataset (tabular).

    Returns:
        X: np.ndarray of shape (n_samples, n_features) float32
        y: np.ndarray of shape (n_samples,) int64
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=seed,
        flip_y=0.05,
    )
    return X.astype(np.float32), y.astype(np.int64)


def get_structured_datasets(
    n_samples: int = STRUCTURED_NUM_SAMPLES,
    n_features: int = STRUCTURED_NUM_FEATURES,
    n_classes: int = STRUCTURED_NUM_CLASSES,
    seed: int = RANDOM_SEED,
    train_ratio: float = 0.8,
) -> tuple:
    """
    Generate structured data and split into train/test TensorDatasets.

    Returns:
        train_dataset, test_dataset, metadata dict
    """
    X, y = generate_structured_data(n_samples, n_features, n_classes, seed)

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    metadata = {
        "data_type": "structured",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return train_ds, test_ds, metadata


# ---------------------------------------------------------------------------
# Unstructured (Image) Data
# ---------------------------------------------------------------------------

def generate_unstructured_data(
    n_samples: int = UNSTRUCTURED_NUM_SAMPLES,
    image_size: tuple = UNSTRUCTURED_IMAGE_SIZE,
    n_classes: int = UNSTRUCTURED_NUM_CLASSES,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Generate synthetic grayscale images with class-dependent patterns.

    Each class gets a unique spatial frequency pattern so the CNN has
    learnable signal, making benchmarks meaningful.

    Returns:
        images: np.ndarray of shape (n_samples, C, H, W) float32
        labels: np.ndarray of shape (n_samples,) int64
    """
    rng = np.random.RandomState(seed)
    C, H, W = image_size
    images = np.zeros((n_samples, C, H, W), dtype=np.float32)
    labels = rng.randint(0, n_classes, size=n_samples).astype(np.int64)

    # Create class-specific frequency patterns
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        count = mask.sum()
        if count == 0:
            continue

        # Deterministic pattern per class: sinusoidal + noise
        freq_x = (cls_idx + 1) * 0.5
        freq_y = (cls_idx + 2) * 0.3
        xx, yy = np.meshgrid(
            np.linspace(0, 2 * np.pi * freq_x, W),
            np.linspace(0, 2 * np.pi * freq_y, H),
        )
        pattern = (np.sin(xx) * np.cos(yy)).astype(np.float32)  # (H, W)

        # Broadcast pattern and add per-sample noise
        noise = rng.randn(count, C, H, W).astype(np.float32) * 0.2
        images[mask] = pattern[np.newaxis, np.newaxis, :, :] + noise

    # Normalize to [0, 1]
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)

    return images, labels


def get_unstructured_datasets(
    n_samples: int = UNSTRUCTURED_NUM_SAMPLES,
    image_size: tuple = UNSTRUCTURED_IMAGE_SIZE,
    n_classes: int = UNSTRUCTURED_NUM_CLASSES,
    seed: int = RANDOM_SEED,
    train_ratio: float = 0.8,
) -> tuple:
    """
    Generate unstructured image data and split into train/test TensorDatasets.

    Returns:
        train_dataset, test_dataset, metadata dict
    """
    images, labels = generate_unstructured_data(n_samples, image_size, n_classes, seed)

    split_idx = int(len(images) * train_ratio)
    X_train, X_test = images[:split_idx], images[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    metadata = {
        "data_type": "unstructured",
        "n_samples": n_samples,
        "image_size": image_size,
        "n_classes": n_classes,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return train_ds, test_ds, metadata


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: TensorDataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator_seed: int = RANDOM_SEED,
) -> DataLoader:
    """
    Create a DataLoader with deterministic shuffling.
    """
    g = torch.Generator()
    g.manual_seed(generator_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )


def _worker_init_fn(worker_id: int):
    """Ensure each DataLoader worker has a deterministic seed."""
    np.random.seed(RANDOM_SEED + worker_id)


# ---------------------------------------------------------------------------
# Convenience: get both datasets with loaders
# ---------------------------------------------------------------------------

def get_all_data(batch_size: int = BATCH_SIZE, pin_memory: bool = False):
    """
    Generate all data and return loaders + metadata for both structured
    and unstructured tasks.

    Returns:
        dict with keys: structured, unstructured
        Each value is a dict with train_loader, test_loader, metadata
    """
    seed_everything(RANDOM_SEED)

    # Structured
    s_train_ds, s_test_ds, s_meta = get_structured_datasets()
    s_train_loader = make_dataloader(s_train_ds, batch_size, shuffle=True, pin_memory=pin_memory)
    s_test_loader = make_dataloader(s_test_ds, batch_size, shuffle=False, pin_memory=pin_memory)

    # Unstructured
    u_train_ds, u_test_ds, u_meta = get_unstructured_datasets()
    u_train_loader = make_dataloader(u_train_ds, batch_size, shuffle=True, pin_memory=pin_memory)
    u_test_loader = make_dataloader(u_test_ds, batch_size, shuffle=False, pin_memory=pin_memory)

    return {
        "structured": {
            "train_loader": s_train_loader,
            "test_loader": s_test_loader,
            "train_dataset": s_train_ds,
            "test_dataset": s_test_ds,
            "metadata": s_meta,
        },
        "unstructured": {
            "train_loader": u_train_loader,
            "test_loader": u_test_loader,
            "train_dataset": u_train_ds,
            "test_dataset": u_test_ds,
            "metadata": u_meta,
        },
    }
