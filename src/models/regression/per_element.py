"""Per-element regression: one small model per element with overlap-aware features."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .peak_features import get_multi_line_features
from .metrics import evaluate_all


def build_neighbor_map(element_names: List[str], overlap_radius: int = 10) -> dict:
    """
    For each element, return the list of elements (including itself) whose
    emission lines fall within ±overlap_radius bins of any of its own lines.

    This defines the minimal feature set needed for each element's sub-model:
    it only needs to see peaks that could physically interfere with its signal.
    """
    line_list = get_multi_line_features(element_names)

    # Group lines by element
    elem_bins: dict[str, list[int]] = {e: [] for e in element_names}
    for elem, _, bin_idx in line_list:
        elem_bins[elem].append(bin_idx)

    neighbor_map = {}
    for elem in element_names:
        my_bins = set(elem_bins[elem])
        neighbors = []
        for other in element_names:
            other_bins = elem_bins[other]
            if any(abs(b - mb) <= overlap_radius for b in other_bins for mb in my_bins):
                neighbors.append(other)
        neighbor_map[elem] = neighbors

    return neighbor_map


def build_per_element_features(
    X_peaks_multi: np.ndarray,
    clf_probs: Optional[np.ndarray],
    neighbor_map: dict,
    element_names: List[str],
    elem_idx: int,
    line_list: list,
) -> np.ndarray:
    """
    Build the feature matrix for element elem_idx.

    Features:
    - Multi-line peak integrals for the element + its overlapping neighbors
    - Ratio-normalised versions of the same
    - Classifier probabilities for the element + its overlapping neighbors (if provided)
    """
    elem = element_names[elem_idx]
    neighbors = neighbor_map[elem]
    neighbor_set = set(neighbors)

    # Which columns in X_peaks_multi correspond to neighbor elements?
    peak_cols = [i for i, (e, _, _) in enumerate(line_list) if e in neighbor_set]

    abs_feats = X_peaks_multi[:, peak_cols]
    totals = X_peaks_multi.sum(axis=1, keepdims=True) + 1e-8
    ratio_feats = abs_feats / totals

    parts = [abs_feats, ratio_feats]

    if clf_probs is not None:
        neighbor_indices = [element_names.index(n) for n in neighbors]
        parts.append(clf_probs[:, neighbor_indices])

    return np.hstack(parts).astype(np.float32)


class PerElementRidgeRegressor:
    """
    41 Ridge regressors, each trained on element-specific overlap-aware features.

    Each sub-model predicts a raw scalar for its element; outputs are clipped to
    [0, ∞) and renormalised to sum to 1 across all elements.
    """
    def __init__(self, alpha: float = 1.0, overlap_radius: int = 10):
        self.alpha = alpha
        self.overlap_radius = overlap_radius
        self._models: list = []
        self._scalers: list = []
        self._neighbor_map: dict = {}
        self._line_list: list = []
        self._element_names: list = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        element_names: List[str],
        clf_probs: Optional[np.ndarray] = None,
    ):
        self._element_names = element_names
        self._line_list = get_multi_line_features(element_names)
        self._neighbor_map = build_neighbor_map(element_names, self.overlap_radius)
        X_peaks = _extract_multi(X, element_names)

        self._models, self._scalers = [], []
        for i in range(len(element_names)):
            feats = build_per_element_features(
                X_peaks, clf_probs, self._neighbor_map, element_names, i, self._line_list
            )
            scaler = StandardScaler()
            feats_s = scaler.fit_transform(feats)
            m = Ridge(alpha=self.alpha)
            m.fit(feats_s, y[:, i])
            self._models.append(m)
            self._scalers.append(scaler)
        return self

    def predict(
        self, X: np.ndarray, clf_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X_peaks = _extract_multi(X, self._element_names)
        preds = np.zeros((X.shape[0], len(self._element_names)), dtype=np.float32)
        for i in range(len(self._element_names)):
            feats = build_per_element_features(
                X_peaks, clf_probs, self._neighbor_map,
                self._element_names, i, self._line_list
            )
            feats_s = self._scalers[i].transform(feats)
            preds[:, i] = self._models[i].predict(feats_s)

        preds = np.clip(preds, 0, None)
        totals = preds.sum(axis=1, keepdims=True)
        totals = np.where(totals < 1e-8, 1.0, totals)
        return preds / totals

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 clf_probs: Optional[np.ndarray] = None) -> dict:
        return evaluate_all(self.predict(X, clf_probs), y,
                            element_names=self._element_names)


def _extract_multi(X: np.ndarray, element_names: List[str]) -> np.ndarray:
    """Extract multi-line absolute peak integrals (no ratios)."""
    from .peak_features import get_multi_line_features
    line_list = get_multi_line_features(element_names)
    n_bins = X.shape[1]
    feats = np.zeros((X.shape[0], len(line_list)), dtype=np.float32)
    for i, (_, _, bin_idx) in enumerate(line_list):
        lo = max(0, bin_idx - 5)
        hi = min(n_bins, bin_idx + 6)
        feats[:, i] = X[:, lo:hi].sum(axis=1)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Per-element CNN
# ─────────────────────────────────────────────────────────────────────────────

def _build_patch_indices(element_names: List[str],
                          neighbor_map: dict,
                          patch_half: int = 10,
                          max_patches: int = 8,
                          n_bins: int = 600) -> List[np.ndarray]:
    """
    For each element, collect the bin indices of all spectral patches
    (own lines + neighbor lines) padded/truncated to max_patches patches.
    Returns a list of (max_patches, 2*patch_half+1) index arrays.
    """
    line_list = get_multi_line_features(element_names)
    elem_lines: dict = {e: [] for e in element_names}
    for elem, _, b in line_list:
        elem_lines[elem].append(b)

    patch_w = 2 * patch_half + 1
    all_patches = []

    for elem in element_names:
        neighbors = neighbor_map[elem]
        # Collect all line bin centres for this element + its neighbors
        bins = []
        # Own lines first
        for b in elem_lines[elem]:
            bins.append(b)
        # Neighbor lines
        for n in neighbors:
            if n != elem:
                for b in elem_lines[n]:
                    bins.append(b)

        # De-duplicate and truncate to max_patches
        seen = []
        for b in bins:
            if b not in seen:
                seen.append(b)
            if len(seen) == max_patches:
                break

        # Build index array: for each patch, indices into spectrum
        idx = np.zeros((max_patches, patch_w), dtype=np.int32)
        for p, b in enumerate(seen):
            for k in range(patch_w):
                idx[p, k] = np.clip(b - patch_half + k, 0, n_bins - 1)
        all_patches.append(idx)  # shape (max_patches, patch_w)

    return all_patches


class PerElementCNNRegressor(nn.Module):
    """
    41 tiny 1D CNNs sharing the same architecture but independent weights.
    Each element's CNN receives a fixed set of spectral patches centred on
    its own emission lines and those of its overlapping neighbors.

    The CNN learns peak shape features (height, width, asymmetry) that linear
    models cannot capture.  Optionally takes classifier probability features
    concatenated to the FC head input.

    Input:
        X_spec   : (batch, 600)  — raw spectrum
        clf_probs: (batch, 41)   — optional classifier sigmoid outputs

    Output: (batch, 41) concentrations summing to 1
    """
    def __init__(self,
                 element_names: List[str],
                 neighbor_map: dict,
                 patch_half: int = 10,
                 max_patches: int = 8,
                 n_bins: int = 600,
                 use_clf: bool = False):
        super().__init__()
        self.element_names = element_names
        self.n_elements = len(element_names)
        self.patch_half = patch_half
        self.max_patches = max_patches
        self.use_clf = use_clf
        patch_w = 2 * patch_half + 1

        # Register patch index arrays as buffers (moved to device automatically)
        patch_indices = _build_patch_indices(
            element_names, neighbor_map, patch_half, max_patches, n_bins
        )
        # Stack to (n_elements, max_patches, patch_w)
        idx_tensor = torch.tensor(np.stack(patch_indices), dtype=torch.long)
        self.register_buffer("patch_idx", idx_tensor)

        # Tiny shared-architecture CNNs (independent weights via ModuleList)
        # Input per element: (batch, 1, max_patches * patch_w)
        cnn_input_len = max_patches * patch_w   # 8 × 21 = 168
        # When use_clf=True, all 41 classifier probs are concatenated to every head.
        # Knowing all elements' detection probabilities helps any element's quantification.
        n_clf = len(element_names) if use_clf else 0

        def make_cnn():
            return nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16), nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(8),   # → (batch, 32, 8)
                nn.Flatten(),              # → (batch, 256)
            )

        self.cnns = nn.ModuleList([make_cnn() for _ in range(self.n_elements)])

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 + n_clf, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            for _ in range(self.n_elements)
        ])

    def _extract_patches(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (batch, 600)
        Returns: (n_elements, batch, max_patches * patch_w)
        """
        batch = X.shape[0]
        patch_w = 2 * self.patch_half + 1
        # patch_idx: (n_elements, max_patches, patch_w)
        # Gather: X[:, patch_idx] → (n_elements, batch, max_patches, patch_w)
        idx = self.patch_idx                              # (n_el, max_p, pw)
        idx_flat = idx.view(self.n_elements, -1)          # (n_el, max_p*pw)
        patches = X[:, idx_flat.view(-1)]                 # (batch, n_el*max_p*pw)
        patches = patches.view(batch, self.n_elements,
                               self.max_patches * patch_w)
        return patches.permute(1, 0, 2)                  # (n_el, batch, max_p*pw)

    def forward(self,
                X_spec: torch.Tensor,
                clf_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        patches = self._extract_patches(X_spec)   # (n_el, batch, max_p*pw)
        outputs = []
        for i in range(self.n_elements):
            p = patches[i]                         # (batch, max_p*pw)
            feat = self.cnns[i](p.unsqueeze(1))    # (batch, 256)
            if self.use_clf and clf_probs is not None:
                feat = torch.cat([feat, clf_probs], dim=1)   # all 41 probs
            outputs.append(self.heads[i](feat).squeeze(-1))  # (batch,)

        logits = torch.stack(outputs, dim=1)       # (batch, n_elements)
        alpha = torch.nn.functional.softplus(logits) + 1e-4
        return alpha / alpha.sum(dim=-1, keepdim=True)


def train_per_element_cnn(
    model: PerElementCNNRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clf_probs_train: Optional[np.ndarray] = None,
    clf_probs_val: Optional[np.ndarray] = None,
    lr: float = 1e-3,
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 15,
) -> dict:
    """Train PerElementCNNRegressor with masked MAE loss + early stopping."""
    import copy
    from tqdm import tqdm
    from torch.utils.data import DataLoader, TensorDataset
    from .losses import MaskedMAELoss

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = MaskedMAELoss()

    def _to_tensor(X, clf):
        tensors = [torch.FloatTensor(X)]
        if clf is not None:
            tensors.append(torch.FloatTensor(clf))
        return tensors

    train_tensors = _to_tensor(X_train, clf_probs_train) + [torch.FloatTensor(y_train)]
    val_tensors   = _to_tensor(X_val,   clf_probs_val)   + [torch.FloatTensor(y_val)]

    train_loader = DataLoader(TensorDataset(*train_tensors),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(*val_tensors),
                              batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_weights, no_improve = float("inf"), None, 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X_b = batch[0].to(device)
            clf_b = batch[1].to(device) if clf_probs_train is not None else None
            y_b  = batch[-1].to(device)
            optimizer.zero_grad()
            pred = model(X_b, clf_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_b  = batch[0].to(device)
                clf_b = batch[1].to(device) if clf_probs_val is not None else None
                y_b  = batch[-1].to(device)
                pred = model(X_b, clf_b)
                val_loss += criterion(pred, y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return history


def predict_per_element_cnn(
    model: PerElementCNNRegressor,
    X: np.ndarray,
    clf_probs: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        X_b   = torch.FloatTensor(X[i:i+batch_size]).to(device)
        clf_b = (torch.FloatTensor(clf_probs[i:i+batch_size]).to(device)
                 if clf_probs is not None else None)
        with torch.no_grad():
            preds.append(model(X_b, clf_b).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared-backbone per-element CNN (fewer parameters)
# ─────────────────────────────────────────────────────────────────────────────

class SharedPerElementCNN(nn.Module):
    """
    One shared CNN backbone applied independently to each element's spectral patches,
    followed by per-element linear heads.

    Parameter count: ~22k vs ~700k for 41 independent CNNs.
    The shared backbone acts as a universal "peak recogniser" — elements with
    similar peak shapes share learned features, reducing overfitting on 10k samples.

    Input:  X_spec (batch, 600), clf_probs (batch, 41) optional
    Output: (batch, 41) concentrations summing to 1
    """
    def __init__(self,
                 element_names: List[str],
                 neighbor_map: dict,
                 patch_half: int = 10,
                 max_patches: int = 8,
                 n_bins: int = 600,
                 use_clf: bool = False):
        super().__init__()
        self.element_names = element_names
        self.n_elements = len(element_names)
        self.patch_half = patch_half
        self.max_patches = max_patches
        self.use_clf = use_clf

        patch_indices = _build_patch_indices(
            element_names, neighbor_map, patch_half, max_patches, n_bins
        )
        self.register_buffer("patch_idx",
                             torch.tensor(np.stack(patch_indices), dtype=torch.long))

        patch_w = 2 * patch_half + 1
        cnn_input_len = max_patches * patch_w

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),   # → 512
        )
        feat_dim = 512
        n_clf = len(element_names) if use_clf else 0

        # Lightweight per-element heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim + n_clf, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            for _ in range(self.n_elements)
        ])

    def _extract_patches(self, X: torch.Tensor) -> torch.Tensor:
        patch_w = 2 * self.patch_half + 1
        batch = X.shape[0]
        idx_flat = self.patch_idx.view(self.n_elements, -1)
        patches = X[:, idx_flat.view(-1)].view(batch, self.n_elements,
                                                self.max_patches * patch_w)
        return patches.permute(1, 0, 2)   # (n_el, batch, len)

    def forward(self,
                X_spec: torch.Tensor,
                clf_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        patches = self._extract_patches(X_spec)   # (n_el, batch, len)
        batch = X_spec.shape[0]

        # Process ALL elements through shared backbone in one call
        all_patches = patches.reshape(
            self.n_elements * batch, 1, patches.shape[-1]
        )
        feats = self.backbone(all_patches)          # (n_el*batch, 512)
        feats = feats.view(self.n_elements, batch, -1)  # (n_el, batch, 512)

        outputs = []
        for i in range(self.n_elements):
            f = feats[i]                            # (batch, 512)
            if self.use_clf and clf_probs is not None:
                f = torch.cat([f, clf_probs], dim=1)
            outputs.append(self.heads[i](f).squeeze(-1))

        logits = torch.stack(outputs, dim=1)        # (batch, n_el)
        alpha = torch.nn.functional.softplus(logits) + 1e-4
        return alpha / alpha.sum(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Element Transformer
# ─────────────────────────────────────────────────────────────────────────────

def _build_element_token_features(element_names: List[str]) -> dict:
    """
    Map each element to the indices of its own peak features in the 129-feature
    multi-line vector, so we can build per-element token embeddings.
    """
    line_list = get_multi_line_features(element_names)
    elem_to_cols = {e: [] for e in element_names}
    for col_idx, (elem, _, _) in enumerate(line_list):
        elem_to_cols[elem].append(col_idx)
    return elem_to_cols


class ElementTransformer(nn.Module):
    """
    Transformer where each of the 41 elements is a token.

    Each token combines:
    - Absolute peak integrals for the element's own emission lines
    - Ratio features (peak_i / Σpeaks) — direct physical proxy for concentration
    - Normalised atomic number Z/92 — element identity / physics constant
    - Normalised primary bin index bin/600 — spectral position context
    - Classifier probability for that element (if use_clf)

    Self-attention lets elements learn inter-element dependencies
    (overlap, sum-to-1 constraint, co-occurrence patterns).

    Input:  X_peaks_multi (batch, 129), clf_probs (batch, 41) optional
    Output: (batch, 41) concentrations summing to 1
    """
    def __init__(self,
                 element_names: List[str],
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 use_clf: bool = False):
        super().__init__()
        self.element_names = element_names
        self.n_elements = len(element_names)
        self.use_clf = use_clf

        elem_to_cols = _build_element_token_features(element_names)
        max_lines = max(len(v) for v in elem_to_cols.values())
        self.max_lines = max_lines

        col_idx = -torch.ones(self.n_elements, max_lines, dtype=torch.long)
        for i, elem in enumerate(element_names):
            cols = elem_to_cols[elem]
            col_idx[i, :len(cols)] = torch.tensor(cols)
        self.register_buffer("col_idx", col_idx)

        # Physics-informed static features per element
        from skbeam.core.constants import XrfElement
        from .peak_features import get_peak_bins
        z_norm = torch.tensor(
            [XrfElement(e).Z / 92.0 for e in element_names], dtype=torch.float32
        )
        primary = get_peak_bins(element_names)
        bin_norm = torch.tensor(
            [(primary[e][2] / 600.0) if primary[e] else 0.0 for e in element_names],
            dtype=torch.float32,
        )
        self.register_buffer("z_norm",   z_norm)    # (n_el,)
        self.register_buffer("bin_norm", bin_norm)  # (n_el,)

        # Token input dim: absolute + ratio + Z + bin + optional clf
        in_dim = max_lines * 2 + 2 + (1 if use_clf else 0)
        self.input_proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self,
                X_peaks_multi: torch.Tensor,
                clf_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = X_peaks_multi.shape[0]
        n_el  = self.n_elements
        dev   = X_peaks_multi.device

        # Absolute peak features: (batch, n_el, max_lines)
        abs_tokens = torch.zeros(batch, n_el, self.max_lines, device=dev)
        for i in range(n_el):
            for j, col in enumerate(self.col_idx[i]):
                if col >= 0:
                    abs_tokens[:, i, j] = X_peaks_multi[:, col]

        # Ratio features: normalise by total
        totals = abs_tokens.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        ratio_tokens = abs_tokens / totals   # (batch, n_el, max_lines)

        # Physics encoding: (batch, n_el, 1) each
        z_feat   = self.z_norm.view(1, n_el, 1).expand(batch, -1, -1)
        bin_feat = self.bin_norm.view(1, n_el, 1).expand(batch, -1, -1)

        parts = [abs_tokens, ratio_tokens, z_feat, bin_feat]
        if self.use_clf and clf_probs is not None:
            parts.append(clf_probs.unsqueeze(-1))   # (batch, n_el, 1)

        tokens = torch.cat(parts, dim=-1)    # (batch, n_el, in_dim)
        x = self.input_proj(tokens)          # (batch, n_el, d_model)
        x = self.transformer(x)              # (batch, n_el, d_model)
        logits = self.output_proj(x).squeeze(-1)   # (batch, n_el)
        alpha = torch.nn.functional.softplus(logits) + 1e-4
        return alpha / alpha.sum(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Element Transformer V2 — with cross-attention to raw spectrum
# ─────────────────────────────────────────────────────────────────────────────

class ElementTransformerV2(nn.Module):
    """
    Extends ElementTransformer with a cross-attention stage where element tokens
    query a CNN-encoded representation of the raw spectrum.

    Stage 1: encode raw spectrum (600) → spatial features (75, d_model) via CNN
    Stage 2: self-attention over element tokens (same as V1)
    Stage 3: cross-attention — element tokens (queries) attend to spectrum features
             (keys/values), learning peak shape and context beyond simple integrals

    Input: X_combined (batch, 600 + 129) — raw spectrum + multi-line peaks concatenated
           clf_probs  (batch, 41) optional
    Output: (batch, 41) concentrations summing to 1
    """
    def __init__(self,
                 element_names: List[str],
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 use_clf: bool = False):
        super().__init__()
        # Reuse V1 for the element token pathway
        self._v1 = ElementTransformer(
            element_names, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, use_clf=use_clf,
        )

        # Small CNN to encode raw spectrum → spatial feature sequence
        self.spec_encoder = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3, stride=2),   # 300
            nn.BatchNorm1d(d_model // 2), nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2, stride=2),  # 150
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),   # 75
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
        )  # output: (batch, d_model, 75)

        # Cross-attention: element tokens query spectrum
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm  = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)   # replaces V1's output_proj

    def forward(self,
                X_combined: torch.Tensor,
                clf_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Split combined input
        X_spec  = X_combined[:, :600]    # raw spectrum
        X_peaks = X_combined[:, 600:]    # 129 multi-line features

        batch = X_spec.shape[0]

        # Encode spectrum to spatial features: (batch, 75, d_model)
        spec_feats = self.spec_encoder(X_spec.unsqueeze(1))   # (batch, d_model, 75)
        spec_feats = spec_feats.permute(0, 2, 1)              # (batch, 75, d_model)

        # Element token pathway (same as V1, up to just before output)
        v1 = self._v1
        n_el = v1.n_elements
        dev  = X_peaks.device

        abs_tokens = torch.zeros(batch, n_el, v1.max_lines, device=dev)
        for i in range(n_el):
            for j, col in enumerate(v1.col_idx[i]):
                if col >= 0:
                    abs_tokens[:, i, j] = X_peaks[:, col]

        totals = abs_tokens.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        ratio_tokens = abs_tokens / totals

        z_feat   = v1.z_norm.view(1, n_el, 1).expand(batch, -1, -1)
        bin_feat = v1.bin_norm.view(1, n_el, 1).expand(batch, -1, -1)
        parts = [abs_tokens, ratio_tokens, z_feat, bin_feat]
        if v1.use_clf and clf_probs is not None:
            parts.append(clf_probs.unsqueeze(-1))

        tokens = torch.cat(parts, dim=-1)
        x = v1.input_proj(tokens)         # (batch, n_el, d_model)
        x = v1.transformer(x)             # (batch, n_el, d_model) — self-attention

        # Cross-attention: element tokens query spectrum
        attended, _ = self.cross_attn(x, spec_feats, spec_feats)
        x = self.cross_norm(x + attended) # residual + norm

        logits = self.output_proj(x).squeeze(-1)   # (batch, n_el)
        alpha = torch.nn.functional.softplus(logits) + 1e-4
        return alpha / alpha.sum(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hard-mask post-processing
# ─────────────────────────────────────────────────────────────────────────────

def apply_hard_mask(preds: np.ndarray,
                    clf_probs: np.ndarray,
                    threshold: float = 0.05) -> np.ndarray:
    """
    Zero out elements the classifier is confident are absent (prob < threshold),
    then renormalise to sum to 1.

    Works as a post-processing step on any model's predictions.
    """
    masked = preds.copy()
    masked[clf_probs < threshold] = 0.0
    totals = masked.sum(axis=1, keepdims=True)
    totals = np.where(totals < 1e-8, 1.0, totals)
    return masked / totals


# ─────────────────────────────────────────────────────────────────────────────
# Generic training/prediction for any nn.Module that takes (X_spec, clf_probs)
# ─────────────────────────────────────────────────────────────────────────────

def train_generic(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    transform_fn=None,
    clf_probs_train=None,
    clf_probs_val=None,
    lr: float = 1e-3,
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 15,
    mixup_alpha: float = 0.0,
    curriculum: bool = False,
) -> dict:
    """Shared training loop for SharedPerElementCNN and ElementTransformer."""
    import copy
    from tqdm import tqdm
    from torch.utils.data import DataLoader, TensorDataset
    from .losses import MaskedMAELoss

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = MaskedMAELoss()

    def prep(X, clf):
        Xi = transform_fn(X) if transform_fn else X
        tensors = [torch.FloatTensor(Xi)]
        if clf is not None:
            tensors.append(torch.FloatTensor(clf))
        return tensors

    tr = prep(X_train, clf_probs_train) + [torch.FloatTensor(y_train)]
    va = prep(X_val,   clf_probs_val)   + [torch.FloatTensor(y_val)]
    val_loader = DataLoader(TensorDataset(*va), batch_size=batch_size, shuffle=False)

    # Curriculum: sort samples by difficulty (number of active elements)
    if curriculum:
        k_per_sample = (y_train > 0).sum(axis=1)   # (N,) — number of active elements
        # Phase boundaries: up to k=2, up to k=3, all
        phases = [
            np.where(k_per_sample <= 2)[0],
            np.where(k_per_sample <= 3)[0],
            np.arange(len(y_train)),
        ]
        phase_epoch_ends = [epochs // 4, epochs // 2, epochs]
    else:
        phases = [np.arange(len(y_train))]
        phase_epoch_ends = [epochs]

    def make_loader(indices):
        subset = [t[indices] for t in tr]
        return DataLoader(TensorDataset(*subset), batch_size=batch_size, shuffle=True)

    train_loader = make_loader(phases[0])

    history = {"train_loss": [], "val_loss": []}
    best_val, best_weights, no_improve = float("inf"), None, 0
    has_clf = clf_probs_train is not None
    phase_idx = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        if curriculum and phase_idx < len(phases) - 1:
            if epoch >= phase_epoch_ends[phase_idx]:
                phase_idx += 1
                train_loader = make_loader(phases[phase_idx])
                print(f"\n[curriculum] Phase {phase_idx+1}: "
                      f"{len(phases[phase_idx])} samples")
        model.train()
        train_batch_losses = []
        for batch in train_loader:
            X_b  = batch[0].to(device)
            clf_b = batch[1].to(device) if has_clf else None
            y_b  = batch[-1].to(device)
            if mixup_alpha > 0:
                import numpy as _np
                lam = float(_np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(X_b.size(0), device=device)
                X_b = lam * X_b + (1 - lam) * X_b[idx]
                y_b = lam * y_b + (1 - lam) * y_b[idx]
                if clf_b is not None:
                    clf_b = lam * clf_b + (1 - lam) * clf_b[idx]
            optimizer.zero_grad()
            pred = model(X_b, clf_b)
            loss = criterion(pred, y_b)
            loss.backward(); optimizer.step()
            train_batch_losses.append(loss.item())
        train_loss = float(np.mean(train_batch_losses))

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_b  = batch[0].to(device)
                clf_b = batch[1].to(device) if has_clf else None
                y_b  = batch[-1].to(device)
                val_batch_losses.append(criterion(model(X_b, clf_b), y_b).item())
        val_loss = float(np.mean(val_batch_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss; best_weights = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}"); break

    if best_weights:
        model.load_state_dict(best_weights)
    return history


def predict_generic(
    model: nn.Module,
    X: np.ndarray,
    transform_fn=None,
    clf_probs: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        Xi = transform_fn(X[i:i+batch_size]) if transform_fn else X[i:i+batch_size]
        X_b  = torch.FloatTensor(Xi).to(device)
        clf_b = (torch.FloatTensor(clf_probs[i:i+batch_size]).to(device)
                 if clf_probs is not None else None)
        with torch.no_grad():
            preds.append(model(X_b, clf_b).cpu().numpy())
    return np.concatenate(preds, axis=0)
