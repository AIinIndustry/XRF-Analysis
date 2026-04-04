import torch
import torch.nn as torch_nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import copy

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


class DenoisingTrainer:
    def __init__(
        self,
        model: torch_nn.Module,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # MSE is standard for denoising tasks, reduction='none' allows for masking
        self.criterion = torch_nn.MSELoss(reduction='none')

    def train(
        self,
        noisy_train: np.ndarray,
        clean_train: np.ndarray,
        noisy_val: np.ndarray,
        clean_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        patience: int = 10,
        min_delta: float = 1e-4,
        train_mask: np.ndarray = None,
        val_mask: np.ndarray = None
    ) -> Dict[str, list]:
        """
        Trains the model with Early Stopping.
        Returns a dictionary containing training and validation loss history.
        """
        if train_mask is not None:
            train_dataset = TensorDataset(torch.FloatTensor(noisy_train), torch.FloatTensor(clean_train), torch.FloatTensor(train_mask))
        else:
            train_dataset = TensorDataset(torch.FloatTensor(noisy_train), torch.FloatTensor(clean_train))
            
        if val_mask is not None:
            val_dataset = TensorDataset(torch.FloatTensor(noisy_val), torch.FloatTensor(clean_val), torch.FloatTensor(val_mask))
        else:
            val_dataset = TensorDataset(torch.FloatTensor(noisy_val), torch.FloatTensor(clean_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                noisy_batch = batch[0].to(self.device)
                clean_batch = batch[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(noisy_batch)
                loss = self.criterion(outputs, clean_batch)
                
                if len(batch) == 3:
                    mask_batch = batch[2].to(self.device)
                    loss = (loss * mask_batch).sum() / (mask_batch.sum() + 1e-8)
                else:
                    loss = loss.mean()
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * noisy_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    noisy_batch = batch[0].to(self.device)
                    clean_batch = batch[1].to(self.device)
                    
                    outputs = self.model(noisy_batch)
                    loss = self.criterion(outputs, clean_batch)
                    
                    if len(batch) == 3:
                        mask_batch = batch[2].to(self.device)
                        loss = (loss * mask_batch).sum() / (mask_batch.sum() + 1e-8)
                    else:
                        loss = loss.mean()
                        
                    val_loss += loss.item() * noisy_batch.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Update pbar
            pbar.set_postfix({"train_loss": f"{train_loss:.2e}", "val_loss": f"{val_loss:.2e}"})

            # Early Stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        if early_stopping.best_model_weights is not None:
            self.model.load_state_dict(early_stopping.best_model_weights)

        return history

    def predict(self, noisy_data: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Applies the trained denoising model to the data.
        """
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(noisy_data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                noisy_batch = batch[0].to(self.device)
                outputs = self.model(noisy_batch)
                predictions.append(outputs.cpu().numpy())
                
        return np.concatenate(predictions, axis=0)

    def evaluate_metrics(self, noisy_data: np.ndarray, clean_data: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluates the model on test data using MSE and MAE.
        """
        preds = self.predict(noisy_data)
        
        if mask is not None:
            mse = np.sum(((preds - clean_data) * mask)**2) / (np.sum(mask) + 1e-8)
            mae = np.sum(np.abs((preds - clean_data) * mask)) / (np.sum(mask) + 1e-8)
        else:
            mse = np.mean((preds - clean_data)**2)
            mae = np.mean(np.abs(preds - clean_data))
        
        return {"MSE": mse, "MAE": mae}