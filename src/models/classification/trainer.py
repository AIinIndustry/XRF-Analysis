import torch
import torch.nn as torch_nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import copy
from sklearn.metrics import f1_score, precision_score, recall_score

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


class ClassificationTrainer:
    def __init__(
        self,
        model: torch_nn.Module,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.criterion = torch_nn.BCELoss()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> Dict[str, list]:
 
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                        
                    val_loss += loss.item() * batch_X.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if early_stopping.best_model_weights is not None:
            self.model.load_state_dict(early_stopping.best_model_weights)

        return history

    def predict(self, X_data: np.ndarray, batch_size: int = 64, threshold: float = 0.5) -> np.ndarray:
       
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X_data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
                
        raw_probs = np.concatenate(predictions, axis=0)
        
        binary_preds = (raw_probs > threshold).astype(int)
        return binary_preds

    def evaluate_metrics(self, X_data: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        
        y_pred = self.predict(X_data, threshold=threshold)
        
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {"F1-Score": f1, "Precision": precision, "Recall": recall}