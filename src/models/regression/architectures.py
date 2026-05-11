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
    def __init__(self, classifier: torch_nn.Module, regressor: torch_nn.Module,
                 threshold: float = 0.5, soft_mask: bool = True):
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold
        # soft_mask=True: use raw sigmoid probabilities as weights instead of
        # hard 0/1 threshold. Prevents unrecoverable false negatives where a
        # present element gets completely zeroed out before renormalisation.
        self.soft_mask = soft_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = self.classifier(x)             # (batch, 41) — sigmoid probabilities
            mask = probs if self.soft_mask else (probs >= self.threshold).float()

        raw_concs = self.regressor(x)              # (batch, 41) — softmax concentrations

        masked = raw_concs * mask
        totals = masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return masked / totals


class CNNRegressorV2(torch_nn.Module):
    """
    Improved 1D CNN regressor addressing three weaknesses of CNNRegressor:

    1. AvgPool instead of MaxPool — preserves peak amplitude (concentration ∝
       peak area, not peak maximum).
    2. No Global Average Pooling — the spatial feature map is kept partially
       intact (reduced to 15 positions) before flattening, so the head retains
       positional information about which energy bins contributed.
    3. Softmax output — avoids the ReLU+normalise collapse where the model
       produces near-uniform predictions when uncertain early in training.

    Input:  (batch, 600)
    Output: (batch, 41)  — concentrations summing to 1
    """
    def __init__(self, input_dim: int = 600, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()
        self.backbone = torch_nn.Sequential(
            # Block 1: 600 → 300
            torch_nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False),
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(2),

            # Block 2: 300 → 150
            torch_nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(2),

            # Block 3: 150 → 75
            torch_nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(2),

            # Block 4: 75 → 15  (retain spatial structure, don't collapse to 1)
            torch_nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(5),
        )
        # 256 channels × 15 positions = 3840
        self.head = torch_nn.Sequential(
            torch_nn.Flatten(),
            torch_nn.Linear(256 * 15, 512),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(512, 128),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(128, n_elements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.head(x)
        return torch.softmax(x, dim=-1)


class SEBlock1D(torch_nn.Module):
    """Squeeze-and-Excitation block: recalibrates channel importance."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = torch_nn.Sequential(
            torch_nn.AdaptiveAvgPool1d(1),
            torch_nn.Flatten(),
            torch_nn.Linear(channels, channels // reduction, bias=False),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(channels // reduction, channels, bias=False),
            torch_nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1)   # (batch, channels, 1)
        return x * scale


class ResConvBlock1D(torch_nn.Module):
    """Conv block with residual connection and optional channel projection."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = torch_nn.Sequential(
            torch_nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1,
                            stride=stride, bias=False),
            torch_nn.BatchNorm1d(out_ch),
            torch_nn.ReLU(inplace=True),
            torch_nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            torch_nn.Sequential(
                torch_nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                torch_nn.BatchNorm1d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else torch_nn.Identity()
        )
        self.relu = torch_nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class CNNRegressorV3(torch_nn.Module):
    """
    V3 improvements over V2:
    - Residual connections throughout the backbone (better gradient flow)
    - SE attention block after final conv (channel recalibration)
    - Wider head: 3840 → 1024 → 256 → 41 (V2 was 512 → 128)
    - Trained with MaskedMAELoss + ReduceLROnPlateau + patience=20

    Input:  (batch, 600)
    Output: (batch, 41)  — concentrations summing to 1
    """
    def __init__(self, input_dim: int = 600, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()
        self.backbone = torch_nn.Sequential(
            # 600 → 300
            ResConvBlock1D(1,   64,  stride=2),
            torch_nn.AvgPool1d(1),          # no-op, keeps interface clean

            # 300 → 150
            ResConvBlock1D(64,  128, stride=2),

            # 150 → 75
            ResConvBlock1D(128, 256, stride=2),

            # 75 → 15
            ResConvBlock1D(256, 256, stride=1),
            torch_nn.AvgPool1d(5),

            # channel attention
            SEBlock1D(256),
        )
        self.head = torch_nn.Sequential(
            torch_nn.Flatten(),              # 256 * 15 = 3840
            torch_nn.Linear(256 * 15, 1024),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(1024, 256),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout / 2),
            torch_nn.Linear(256, n_elements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.head(x)
        return torch.softmax(x, dim=-1)


class CrossAttentionHead(torch_nn.Module):
    """
    Cross-attention head for per-element concentration prediction.

    41 learnable element query embeddings attend over the spatial CNN feature
    map. Each element gets its own weighted aggregation of spectral features,
    which is more expressive than a shared flat FC head.

    queries  : (batch, 41, d_model)   — one per element
    keys/vals: (batch, n_spatial, d_model) — from CNN backbone
    """
    def __init__(self, feature_dim: int = 256, n_spatial: int = 15,
                 n_elements: int = 41, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.element_queries = torch_nn.Parameter(torch.randn(n_elements, d_model))
        self.feat_proj = torch_nn.Linear(feature_dim, d_model)
        self.attn = torch_nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_proj = torch_nn.Linear(d_model, 1)
        torch_nn.init.xavier_uniform_(self.element_queries.data.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, feature_dim, n_spatial)
        batch = x.shape[0]
        kv = self.feat_proj(x.permute(0, 2, 1))                         # (batch, n_spatial, d_model)
        q  = self.element_queries.unsqueeze(0).expand(batch, -1, -1)    # (batch, 41, d_model)
        attended, _ = self.attn(q, kv, kv)                              # (batch, 41, d_model)
        return self.out_proj(attended).squeeze(-1)                       # (batch, 41) logits


class CNNRegressorV4(torch_nn.Module):
    """
    V4 improvements over V3:
    - Cross-attention head: each element has a learnable query that attends
      over the 15 spatial CNN positions, giving per-element feature aggregation
      instead of a shared flat FC head.
    - Dirichlet NLL loss: outputs raw logits; concentrations are the mean of
      the implied Dirichlet (alpha / sum(alpha)).

    Call forward() for predictions (concentrations).
    Call forward_logits() to get raw logits for DirichletNLLLoss.

    Input:  (batch, 600)
    Output: (batch, 41)  — concentrations summing to 1
    """
    def __init__(self, input_dim: int = 600, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()
        self.backbone = torch_nn.Sequential(
            ResConvBlock1D(1,   64,  stride=2),   # 600 → 300
            ResConvBlock1D(64,  128, stride=2),   # 300 → 150
            ResConvBlock1D(128, 256, stride=2),   # 150 → 75
            ResConvBlock1D(256, 256, stride=1),
            torch_nn.AvgPool1d(5),                # 75 → 15
            SEBlock1D(256),
        )
        self.dropout = torch_nn.Dropout(dropout)
        self.head = CrossAttentionHead(feature_dim=256, n_spatial=15,
                                       n_elements=n_elements, d_model=64, n_heads=4)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.dropout(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        alpha = F.softplus(logits) + 1.0
        return alpha / alpha.sum(dim=-1, keepdim=True)


class MLPRegressor(torch_nn.Module):
    """
    Wide MLP regressor. Since PLS (a linear model) is already competitive with
    the CNN on this task, a wide MLP with BatchNorm can be a strong baseline
    without the spatial-information loss introduced by pooling operations.

    Input:  (batch, 600)
    Output: (batch, 41)  — concentrations summing to 1
    """
    def __init__(self, input_dim: int = 600, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, 1024),
            torch_nn.BatchNorm1d(1024),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),

            torch_nn.Linear(1024, 512),
            torch_nn.BatchNorm1d(512),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),

            torch_nn.Linear(512, 256),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout / 2),

            torch_nn.Linear(256, 128),
            torch_nn.ReLU(inplace=True),

            torch_nn.Linear(128, n_elements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class PhysicsInformedRegressor(torch_nn.Module):
    """
    Dual-branch regressor using physics-informed peak features.

    Branch A — CNN on the raw spectrum (600 bins): learns contextual features
               like background shape and peak overlap patterns.
    Branch B — MLP on 41 peak integrals (one per element at its characteristic
               emission line): encodes direct physical prior knowledge that
               concentration ∝ peak area.

    The two branches are concatenated and fed to a shared output head.

    Input:  (batch, 641)  — first 600 dims = log-spectrum, last 41 = peak integrals
    Output: (batch, 41)   — concentrations summing to 1
    """
    def __init__(self, n_elements: int = 41, dropout: float = 0.3):
        super().__init__()

        # Branch A: 1D CNN on spectrum (first 600 dims)
        self.cnn = torch_nn.Sequential(
            torch_nn.Conv1d(1, 64,  kernel_size=7, padding=3, bias=False),
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(2),                              # → 300

            torch_nn.Conv1d(64,  128, kernel_size=5, padding=2, bias=False),
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(2),                              # → 150

            torch_nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(2),                              # → 75

            torch_nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(inplace=True),
            torch_nn.AvgPool1d(5),                              # → 15
        )
        cnn_out = 256 * 15   # 3840

        # Branch B: MLP on peak integrals (last 41 dims)
        self.peak_mlp = torch_nn.Sequential(
            torch_nn.Linear(n_elements, 128),
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(128, 128),
            torch_nn.ReLU(inplace=True),
        )
        peak_out = 128

        # Fusion head
        self.head = torch_nn.Sequential(
            torch_nn.Linear(cnn_out + peak_out, 512),
            torch_nn.ReLU(inplace=True),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(512, 128),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(128, n_elements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input
        spectrum   = x[:, :600]    # (batch, 600)
        peak_feats = x[:, 600:]    # (batch, 41)

        # CNN branch
        cnn_out = self.cnn(spectrum.unsqueeze(1))  # (batch, 256, 15)
        cnn_out = cnn_out.flatten(1)               # (batch, 3840)

        # Peak MLP branch
        peak_out = self.peak_mlp(peak_feats)       # (batch, 128)

        # Fuse and predict
        fused = torch.cat([cnn_out, peak_out], dim=1)
        out = self.head(fused)
        return torch.softmax(out, dim=-1)
