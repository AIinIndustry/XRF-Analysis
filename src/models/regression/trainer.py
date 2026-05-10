import torch
import torch.nn as torch_nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
import copy

from .losses import CombinedRegressionLoss
from .metrics import evaluate_all
from ..denoising.preprocessing import StandardScaler, LogMinMaxScaler


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss: float, model: torch_nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0


class RegressionTrainer:
    """
    Training loop for elemental concentration regression.

    Mirrors the DenoisingTrainer interface for consistency.

    Args:
        model:          A CNNRegressor or ResNetRegressor instance.
        learning_rate:  Adam learning rate.
        mse_weight:     Weight for the MaskedMSE term in CombinedRegressionLoss.
        kl_weight:      Weight for the KL-divergence term.
        device:         'cuda' or 'cpu'.
    """
    def __init__(
        self,
        model: torch_nn.Module,
        learning_rate: float = 1e-3,
        mse_weight: float = 1.0,
        kl_weight: float = 0.5,
        normalize: bool = True,
        scaler: str = "standard",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = CombinedRegressionLoss(mse_weight=mse_weight, kl_weight=kl_weight)
        self.normalize = normalize
        if normalize:
            if scaler == "log_minmax":
                self.scaler = LogMinMaxScaler()
            else:
                self.scaler = StandardScaler()
        else:
            self.scaler = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> Dict[str, list]:
        """
        Trains the model with early stopping.

        Args:
            X_train:    (N_train, 600) spectra
            y_train:    (N_train, 41) concentration targets (floats, sum to 1)
            X_val:      (N_val, 600)
            y_val:      (N_val, 41)
            epochs:     maximum number of epochs
            batch_size: mini-batch size
            patience:   early stopping patience (epochs without improvement)
            min_delta:  minimum improvement to reset patience counter

        Returns:
            history dict with 'train_loss' and 'val_loss' lists.
        """
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_val   = self.scaler.transform(X_val)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = self.model(X_batch)
                    loss = self.criterion(pred, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            pbar.set_postfix({"train": f"{train_loss:.4e}", "val": f"{val_loss:.4e}"})

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if early_stopping.best_model_weights is not None:
            self.model.load_state_dict(early_stopping.best_model_weights)

        return history

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Returns predicted concentrations for all samples in X.

        Output shape: (N, 41), each row sums to ~1.
        """
        if self.normalize:
            X = self.scaler.transform(X)

        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        element_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Runs predict() and computes all metrics.

        Args:
            X:              (N, 600) spectra
            y:              (N, 41) ground-truth concentrations
            element_names:  list of element symbols for per-element MAE

        Returns:
            dict with 'masked_mae', 'masked_mse', 'masked_r2',
            and optionally 'per_element_mae' (pd.Series).
        """
        pred = self.predict(X)
        return evaluate_all(pred, y, element_names=element_names)
