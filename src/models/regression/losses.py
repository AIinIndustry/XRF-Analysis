import torch
import torch.nn as torch_nn
import torch.nn.functional as F


class MaskedMSELoss(torch_nn.Module):
    """
    MSE loss computed only on elements with non-zero ground-truth concentration.

    Rationale: predicting 0 for absent elements is trivial and would dominate
    a plain MSE, inflating apparent performance. This focuses the loss signal
    on the elements that actually matter.

    Args:
        reduction: 'mean' averages over all active (element, sample) pairs.
                   'sum' sums them (useful for debugging).
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (batch, n_elements)
        mask = (target > 0).float()
        sq_err = (pred - target) ** 2 * mask

        if self.reduction == 'mean':
            return sq_err.sum() / (mask.sum() + 1e-8)
        return sq_err.sum()


class MaskedMAELoss(torch_nn.Module):
    """
    MAE loss computed only on elements with non-zero ground-truth concentration.
    Can be used as an alternative to MaskedMSELoss — less sensitive to outliers.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (target > 0).float()
        abs_err = torch.abs(pred - target) * mask

        if self.reduction == 'mean':
            return abs_err.sum() / (mask.sum() + 1e-8)
        return abs_err.sum()


class KLDivergenceLoss(torch_nn.Module):
    """
    KL divergence treating concentrations as a probability distribution.

    KL(target || pred) = sum(target * log(target / pred))

    This is a natural fit when the output is a Softmax (i.e. a probability
    simplex) and the target is also a distribution (Dirichlet-sampled).
    It penalises placing probability mass in the wrong elements more strongly
    than MSE does.

    Numerically stable: clamps pred and target away from 0.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Both tensors: (batch, n_elements)
        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)
        # Only compute over active elements (avoids 0 * log(0/x) instability)
        mask = (target > self.eps).float()
        kl = target * (torch.log(target) - torch.log(pred)) * mask
        return kl.sum(dim=-1).mean()


class CombinedRegressionLoss(torch_nn.Module):
    """
    Weighted combination of MaskedMSE and KLDivergence.

    MaskedMSE provides direct error signal on concentrations.
    KL-divergence penalises distributional mismatch (wrong element gets the mass).
    Together they make the model sensitive to both magnitude and relative ordering.

    Args:
        mse_weight:  weight for the MaskedMSE term (default 1.0)
        kl_weight:   weight for the KL term (default 0.5)
    """
    def __init__(self, mse_weight: float = 1.0, kl_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.mse = MaskedMSELoss()
        self.kl = KLDivergenceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + self.kl_weight * self.kl(pred, target)
