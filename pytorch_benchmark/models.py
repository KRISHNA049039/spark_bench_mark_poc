"""
PyTorch model definitions for structured (tabular) and unstructured (image) data.

Models:
    - TabularNet: A multi-layer feed-forward network for tabular classification.
    - ImageCNN: A convolutional neural network for grayscale image classification.

Both models support deterministic weight initialization for reproducibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_benchmark.config import (
    STRUCTURED_NUM_FEATURES,
    STRUCTURED_NUM_CLASSES,
    UNSTRUCTURED_IMAGE_SIZE,
    UNSTRUCTURED_NUM_CLASSES,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Weight initialization utilities
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module):
    """
    Deterministic weight initialization using Kaiming uniform for linear/conv
    layers and zeros for biases.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Structured Data Model: TabularNet
# ---------------------------------------------------------------------------

class TabularNet(nn.Module):
    """
    Multi-layer feed-forward network for tabular/structured data classification.

    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x 3 -> Linear -> Output

    This is inspired by simple TabNet-style architectures but kept lightweight
    for benchmarking purposes.
    """

    def __init__(
        self,
        n_features: int = STRUCTURED_NUM_FEATURES,
        n_classes: int = STRUCTURED_NUM_CLASSES,
        hidden_dims: tuple = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        layers = []
        in_dim = n_features

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

        # Deterministic initialization
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) float tensor

        Returns:
            logits: (batch_size, n_classes) float tensor
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Unstructured Data Model: ImageCNN
# ---------------------------------------------------------------------------

class ImageCNN(nn.Module):
    """
    Convolutional neural network for grayscale image classification.

    Architecture:
        Conv2d -> BN -> ReLU -> MaxPool ->
        Conv2d -> BN -> ReLU -> MaxPool ->
        Conv2d -> BN -> ReLU -> AdaptiveAvgPool ->
        Flatten -> Linear -> ReLU -> Dropout -> Linear -> Output

    Designed for 28x28 grayscale images but adapts to other sizes via
    adaptive average pooling.
    """

    def __init__(
        self,
        in_channels: int = None,
        image_size: tuple = UNSTRUCTURED_IMAGE_SIZE,
        n_classes: int = UNSTRUCTURED_NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()

        if in_channels is None:
            in_channels = image_size[0]

        self.n_classes = n_classes
        self.image_size = image_size

        # Convolutional backbone
        self.conv_layers = nn.Sequential(
            # Block 1: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Fully connected head
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        # Deterministic initialization
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, C, H, W) float tensor

        Returns:
            logits: (batch_size, n_classes) float tensor
        """
        features = self.conv_layers(x)
        logits = self.fc_layers(features)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    data_type: str,
    device: torch.device = None,
    seed: int = RANDOM_SEED,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a model for the given data type.

    Args:
        data_type: 'structured' or 'unstructured'
        device: target device (cpu or cuda)
        seed: random seed for weight initialization
        **kwargs: additional model constructor args

    Returns:
        Initialized model on the specified device.
    """
    if device is None:
        device = torch.device("cpu")

    # Ensure deterministic initialization
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if data_type == "structured":
        model = TabularNet(**kwargs)
    elif data_type == "unstructured":
        model = ImageCNN(**kwargs)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'structured' or 'unstructured'.")

    model = model.to(device)
    return model


def get_model_summary(model: nn.Module) -> dict:
    """
    Return a summary of model parameters.

    Returns:
        dict with total_params, trainable_params, non_trainable_params, model_size_mb
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params

    # Model size in MB (assuming float32)
    size_mb = total_params * 4 / (1024 * 1024)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable,
        "model_size_mb": round(size_mb, 4),
    }
