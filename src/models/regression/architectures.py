import torch
import torch.nn as torch_nn
import torch.nn.functional as F


class CNNRegressor(torch_nn.Module):
    """
    1D CNN backbone with a Softmax output head for elemental concentration prediction.

    The Softmax activation enforces the physical constraint that concentrations
    must sum to 1.0 — matching the Dirichlet-distributed targets.

    Input:  (batch, 600)   — XRF spectrum
    Output: (batch, 41)    — concentration per element, summing to 1
    """
    def __init__(self, input_dim: int = 600, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()
        self.backbone = torch_nn.Sequential(
            # Block 1: 600 → 300
            torch_nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False),
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(inplace=True),
            torch_nn.MaxPool1d(2),

            # Block 2: 300 → 150
            torch_nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(inplace=True),
            torch_nn.MaxPool1d(2),

            # Block 3: 150 → 75
            torch_nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
            torch_nn.MaxPool1d(2),
        )
        self.head = torch_nn.Sequential(
            torch_nn.AdaptiveAvgPool1d(1),   # (batch, 256, 1)
            torch_nn.Flatten(),               # (batch, 256)
            torch_nn.Linear(256, 128),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(128, n_elements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.head(x)
        x = F.relu(x)
        return x / (x.sum(dim=-1, keepdim=True) + 1e-8)


class ResidualBlock1D(torch_nn.Module):
    """Basic 1D residual block with two conv layers and a skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = torch_nn.Sequential(
            torch_nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(channels),
            torch_nn.ReLU(inplace=True),
            torch_nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(channels),
        )
        self.relu = torch_nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ResNetRegressor(torch_nn.Module):
    """
    1D ResNet regressor. Uses residual blocks for more stable gradients and
    better feature reuse — particularly useful when spectra have overlapping peaks.

    Input:  (batch, 600)
    Output: (batch, 41)   — concentrations summing to 1
    """
    def __init__(self, input_dim: int = 600, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()
        self.entry = torch_nn.Sequential(
            torch_nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 300
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(inplace=True),
        )
        self.stage1 = torch_nn.Sequential(
            ResidualBlock1D(64),
            torch_nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 150
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(inplace=True),
        )
        self.stage2 = torch_nn.Sequential(
            ResidualBlock1D(128),
            torch_nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 75
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
        )
        self.head = torch_nn.Sequential(
            torch_nn.AdaptiveAvgPool1d(1),
            torch_nn.Flatten(),
            torch_nn.Linear(256, 128),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(128, n_elements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.entry(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.head(x)
        x = F.relu(x)
        return x / (x.sum(dim=-1, keepdim=True) + 1e-8)


class TwoStageRegressor(torch_nn.Module):
    """
    Two-stage model: first classifies which elements are present,
    then estimates concentrations only for the predicted active elements.

    Stage 1 (classifier): spectrum → binary mask (which elements are present)
    Stage 2 (regressor):  spectrum + mask → concentrations

    At inference, the mask from Stage 1 is used to zero out absent elements,
    and the remaining values are renormalized to sum to 1.

    Note: the two stages are trained separately. This class handles inference only.
    See trainer.py for the joint training loop.
    """
    def __init__(self, classifier: torch_nn.Module, regressor: torch_nn.Module, threshold: float = 0.5):
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = self.classifier(x)             # (batch, 41) — sigmoid probabilities
            mask = (probs >= self.threshold).float()

        raw_concs = self.regressor(x)              # (batch, 41) — softmax concentrations

        # Zero out elements predicted as absent, then renormalize
        masked = raw_concs * mask
        totals = masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return masked / totals
