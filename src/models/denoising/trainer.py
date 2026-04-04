import torch
import torch.nn as torch_nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

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
        # MSE is standard for denoising tasks
        self.criterion = torch_nn.MSELoss()

    def train(
        self,
        noisy_train: np.ndarray,
        clean_train: np.ndarray,
        noisy_val: np.ndarray,
        clean_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32
    ) -> Dict[str, list]:
        """
        Trains the model.
        Returns a dictionary containing training and validation loss history.
        """
        train_dataset = TensorDataset(torch.FloatTensor(noisy_train), torch.FloatTensor(clean_train))
        val_dataset = TensorDataset(torch.FloatTensor(noisy_val), torch.FloatTensor(clean_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for noisy_batch, clean_batch in train_loader:
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(noisy_batch)
                loss = self.criterion(outputs, clean_batch)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * noisy_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy_batch, clean_batch in val_loader:
                    noisy_batch = noisy_batch.to(self.device)
                    clean_batch = clean_batch.to(self.device)
                    
                    outputs = self.model(noisy_batch)
                    loss = self.criterion(outputs, clean_batch)
                    val_loss += loss.item() * noisy_batch.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Print occasionally
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

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

    def evaluate_metrics(self, noisy_data: np.ndarray, clean_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the model on test data using MSE and MAE.
        """
        preds = self.predict(noisy_data)
        
        mse = np.mean((preds - clean_data)**2)
        mae = np.mean(np.abs(preds - clean_data))
        
        return {"MSE": mse, "MAE": mae}