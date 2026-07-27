"""
Base runner class with shared training/evaluation logic.

All mode-specific runners inherit from this to ensure consistent
metrics collection and result formatting.
"""

import time
import gc
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_benchmark.config import EPOCHS, LEARNING_RATE, RANDOM_SEED
from pytorch_benchmark.data_generation import seed_everything


class RunnerResult:
    """Container for benchmark results from a single run."""

    def __init__(self, mode: str, data_type: str):
        self.mode = mode
        self.data_type = data_type
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.test_loss: float = 0.0
        self.test_accuracy: float = 0.0
        self.predictions: np.ndarray = None
        self.probabilities: np.ndarray = None
        self.epoch_times: List[float] = []
        self.total_train_time: float = 0.0
        self.total_inference_time: float = 0.0
        self.resource_metrics: Dict[str, Any] = {}
        self.model_state_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "mode": self.mode,
            "data_type": self.data_type,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
            "predictions_hash": _hash_array(self.predictions),
            "epoch_times": self.epoch_times,
            "total_train_time": self.total_train_time,
            "total_inference_time": self.total_inference_time,
            "resource_metrics": self.resource_metrics,
            "model_state_hash": self.model_state_hash,
        }


def _hash_array(arr: np.ndarray) -> str:
    """Create a reproducible hash of a numpy array."""
    if arr is None:
        return ""
    import hashlib
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


class BaseRunner(ABC):
    """
    Abstract base class for all execution mode runners.

    Provides common training loop, evaluation, and metrics collection.
    Subclasses override device placement and resource monitoring hooks.
    """

    def __init__(self, mode: str, seed: int = RANDOM_SEED):
        self.mode = mode
        self.seed = seed
        self.device = self._get_device()

    @abstractmethod
    def _get_device(self) -> torch.device:
        """Return the target device for this runner."""
        ...

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to target device."""
        return tensor.to(self.device, non_blocking=True)

    def _get_optimizer(self, model: nn.Module, lr: float = LEARNING_RATE):
        """Create optimizer."""
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _get_criterion(self):
        """Create loss function."""
        return nn.CrossEntropyLoss()

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Hash model state dict for reproducibility checks."""
        import hashlib
        h = hashlib.sha256()
        for key in sorted(model.state_dict().keys()):
            param = model.state_dict()[key].cpu().numpy()
            h.update(param.tobytes())
        return h.hexdigest()[:32]

    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple:
        """
        Train for one epoch.

        Returns:
            (avg_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = self._move_to_device(batch_x)
            batch_y = self._move_to_device(batch_y)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple:
        """
        Evaluate model on a dataset.

        Returns:
            (avg_loss, accuracy, all_predictions, all_probabilities)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []

        for batch_x, batch_y in loader:
            batch_x = self._move_to_device(batch_x)
            batch_y = self._move_to_device(batch_y)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)

        return avg_loss, accuracy, predictions, probabilities

    def run(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_type: str,
        epochs: int = EPOCHS,
        lr: float = LEARNING_RATE,
        resource_monitor=None,
    ) -> RunnerResult:
        """
        Execute full training + evaluation pipeline.

        Args:
            model: PyTorch model (already on correct device)
            train_loader: training DataLoader
            test_loader: test DataLoader
            data_type: 'structured' or 'unstructured'
            epochs: number of training epochs
            lr: learning rate
            resource_monitor: optional ResourceMonitor instance

        Returns:
            RunnerResult with all metrics
        """
        seed_everything(self.seed)
        result = RunnerResult(self.mode, data_type)

        model = model.to(self.device)
        optimizer = self._get_optimizer(model, lr)
        criterion = self._get_criterion()

        # Start resource monitoring
        if resource_monitor:
            resource_monitor.start()

        # --- Training ---
        train_start = time.perf_counter()

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            loss, acc = self.train_epoch(model, train_loader, optimizer, criterion)
            epoch_time = time.perf_counter() - epoch_start

            result.train_losses.append(loss)
            result.train_accuracies.append(acc)
            result.epoch_times.append(epoch_time)

        result.total_train_time = time.perf_counter() - train_start

        # --- Evaluation ---
        infer_start = time.perf_counter()
        test_loss, test_acc, preds, probs = self.evaluate(model, test_loader, criterion)
        result.total_inference_time = time.perf_counter() - infer_start

        result.test_loss = test_loss
        result.test_accuracy = test_acc
        result.predictions = preds
        result.probabilities = probs

        # Model state hash for reproducibility
        result.model_state_hash = self._compute_model_hash(model)

        # Stop resource monitoring and collect metrics
        if resource_monitor:
            resource_monitor.stop()
            result.resource_metrics = resource_monitor.get_summary()

        # Cleanup
        self._cleanup()

        return result

    def _cleanup(self):
        """Force garbage collection after run."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
