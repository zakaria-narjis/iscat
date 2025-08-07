import torch
from torch import nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, input, target):
        return torch.sqrt(self.mse(input, target))

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss for PyTorch.
    
    The MMD loss measures the distance between two probability distributions
    using kernel methods. Lower values indicate more similar distributions.
    
    Args:
        kernel (str): Kernel type, either "rbf" or "multiscale"
        bandwidth_range (list, optional): Custom bandwidth values for the kernel.
                                        If None, uses default ranges.
    """
    
    def __init__(self, kernel="rbf", bandwidth_range=None):
        super(MMDLoss, self).__init__()
        self.kernel = kernel

        if bandwidth_range is None:
            if kernel == "rbf":
                self.bandwidth_range = [5, 10, 15, 20, 50]
            elif kernel == "multiscale":
                self.bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            else:
                raise ValueError(f"Unknown kernel type: {kernel}")
        else:
            self.bandwidth_range = bandwidth_range
    
    def forward(self, x, y):
        """
        Compute the MMD loss between two samples.
        
        Args:
            x (torch.Tensor): First sample, distribution P
            y (torch.Tensor): Second sample, distribution Q
            
        Returns:
            torch.Tensor: MMD loss value
        """   
        # Ensure samples are 2D
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Compute pairwise distances
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        
        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
        
        # Initialize kernel matrices
        XX = torch.zeros(xx.shape, device=x.device)
        YY = torch.zeros(xx.shape, device=x.device)
        XY = torch.zeros(xx.shape, device=x.device)
        
        # Compute kernel values based on kernel type
        if self.kernel == "multiscale":
            for a in self.bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        elif self.kernel == "rbf":
            for a in self.bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)
        
        return torch.mean(XX + YY - 2. * XY)
    
def distance_matrix(a, b):
    a_expanded = a.view(-1, 1)
    b_expanded = b.view(1, -1)

    return torch.abs(a_expanded - b_expanded)

def mmd_loss(x, y, kernel="rbf"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    random_indices = torch.randperm(x.shape[0])
    y = y[random_indices]
    # sort the samples to ensure that the order does not affect the loss
    x = x.sort()[0]
    y = y.sort()[0]
    if x.dim()== 1:
        x = x.unsqueeze(1)
    if y.dim()== 1:
        y = y.unsqueeze(1)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))
    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [5, 10, 15, 20, 50]  #[10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)

def knnDivergence(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
    k_neighbors: torch.Tensor,
    smoothing_kernel=None,
    reduction="mean",
    method="absolute",
):
    """
    Computes the KNN divergence between two sets of points.
    Args:
        points_x (torch.Tensor): First set of points (shape: [N, D])
        points_y (torch.Tensor): Second set of points (shape: [M, D])
        k_neighbors (torch.Tensor): Number of neighbors to consider
        smoothing_kernel (torch.Tensor, optional): Kernel for smoothing the distances
        reduction (str, optional): Reduction method ('mean', 'sum', 'none')
        method (str, optional): Method for divergence calculation ('fraction', 'absolute')
    Returns:
        torch.Tensor: KNN divergence value
    """
    # Compute the distance matrices
    xx_distances = distance_matrix(points_x, points_x)
    xy_distances = distance_matrix(
        points_x, points_y
    )  # one row for every sample in x, one col for every sample in y

    # if the sets have different sizes
    # e.g. y has twice as many points -> the distance to the 3rd closest point in x should be the same as the distance to the 6th point in y
    k_multiplier = points_y.shape[0] / points_x.shape[0]

    k_dist_xx = torch.sort(xx_distances, dim=1)[0][:, k_neighbors]
    k_dist_xy = torch.sort(xy_distances, dim=1)[0][
        :, (k_neighbors * k_multiplier).to(torch.int)
    ]
    # optional: smoothen the distances
    # (so that it matters less whether a point is the i-th or the (i+1)-th closest neighbor)
    if smoothing_kernel is not None:
        # torch conv1d demands a channel dimension, hence the (un)squeezing
        k_dist_xx = torch.nn.functional.conv1d(
            k_dist_xx.unsqueeze(1), weight=smoothing_kernel.view(1, 1, -1)
        ).flatten(1)
        k_dist_xy = torch.nn.functional.conv1d(
            k_dist_xy.unsqueeze(1), weight=smoothing_kernel.view(1, 1, -1)
        ).flatten(1)
    if method == "fraction":
        # scale-invariant, but trains less easily
        output = (1 - k_dist_xx / k_dist_xy) ** 2
    elif method == "absolute":
        # trains more easily, but not scale-invariant. Can be useful as a first step.
        output = (k_dist_xx - k_dist_xy) ** 2
    else:
        raise ValueError(
            "Invalid method. Choose either 'fraction' or 'absolute'."
        )
    if reduction == "mean":
        return torch.mean(output)
    elif reduction == "sum":
        return torch.sum(output)
    elif reduction == "none":
        return output
    else:
        raise ValueError(
            "Invalid reduction method. Choose either 'mean', 'sum', or 'none'."
        )

def contrast_from_middle_rows(images: torch.Tensor,offset=1) -> torch.Tensor:
    """
    Compute standard deviation (RMS contrast) from the 4 middle rows of each image in a batch.
    
    Args:
        images: Tensor of shape (B, C, H, W)
    
    Returns:
        Tensor of shape (B,) with contrast values.
    """
    B, C, H, W = images.shape
    mid = H // 2
    rows = images[:, :, mid - offset:mid + offset, :]  # (B, C, 4, W)

    gray = rows[:,0,:,:]  # Convert to grayscale: (B, 4, W)
    contrast = gray.std(dim=(1, 2))  # Compute std over height and width

    return contrast  # shape: (B,)

def MonotonicityLoss(
    predictions: torch.Tensor,
    images: torch.Tensor,
    increasing: bool = True,
    rmse: bool = False
) -> torch.Tensor:
    """
    Enforces that predictions follow the same monotonic order as contrast,
    using contrast differences as soft penalty weights.

    Args:
        predictions (torch.Tensor): Predicted values of shape (B,)
        images (torch.Tensor): Batch of images of shape (B, C, H, W)
        increasing (bool): If True, enforce increasing order, else decreasing
        rmse (bool): If True, use RMSE-like loss, else MAE-like loss

    Returns:
        torch.Tensor: Scalar monotonicity loss
    """
    contrast_values = contrast_from_middle_rows(images)  # Shape: (B,)
    predictions = predictions.view(-1)                   # Shape: (B,)
    contrast_values = contrast_values.view(-1)           # Shape: (B,)

    # Compute pairwise differences
    contrast_diff = contrast_values.unsqueeze(0) - contrast_values.unsqueeze(1)  # (B, B)
    pred_diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)              # (B, B)

    if increasing:
        # Penalize when pred_i >= pred_j while contrast_i < contrast_j
        penalty_mask = contrast_diff < 0
        penalty_weight = -contrast_diff * penalty_mask  # make weights positive
        violation = torch.relu(pred_diff) * penalty_weight
    else:
        # Penalize when pred_i <= pred_j while contrast_i < contrast_j
        penalty_mask = contrast_diff < 0
        penalty_weight = -contrast_diff * penalty_mask
        violation = torch.relu(-pred_diff) * penalty_weight

    # Compute final loss
    if rmse:
        loss = (violation ** 2).sum() / (penalty_mask.sum() + 1e-8)
        loss = torch.sqrt(loss)
    else:
        loss = violation.sum() / (penalty_mask.sum() + 1e-8)

    return loss

    
class KNNDivergenceLoss(nn.Module):
    """
    Optimized KNN Divergence Loss for PyTorch with robust edge case handling.
    
    This loss computes the divergence between two point sets based on k-nearest
    neighbor distances, useful for distribution matching and generative modeling.
    Handles edge cases where batch size is smaller than k_neighbors (e.g., last batch).
    
    Args:
        k_neighbors (int): Number of neighbors to consider for divergence calculation
        smoothing_kernel (torch.Tensor, optional): 1D kernel for smoothing distances
        reduction (str): Reduction method - 'mean', 'sum', or 'none'
        method (str): Divergence method - 'fraction' or 'absolute'
        eps (float): Small epsilon for numerical stability
        adaptive_k (bool): Whether to adaptively reduce k when batch size is too small
        min_batch_size (int): Minimum batch size to compute loss (returns 0 if smaller)
    """
    
    def __init__(
        self,
        k_neighbors: int,
        smoothing_kernel: torch.Tensor = None,
        reduction: str = "mean",
        method: str = "absolute",
        eps: float = 1e-8,
        adaptive_k: bool = True,
        min_batch_size: int = 2
    ):
        super(KNNDivergenceLoss, self).__init__()
        
        self.k_neighbors = k_neighbors
        self.reduction = reduction
        self.method = method
        self.eps = eps
        self.adaptive_k = adaptive_k
        self.min_batch_size = min_batch_size
        
        # Register smoothing kernel as buffer if provided
        if smoothing_kernel is not None:
            self.register_buffer('smoothing_kernel', smoothing_kernel.view(1, 1, -1))
        else:
            self.smoothing_kernel = None
            
        # Validate parameters
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if method not in ['fraction', 'absolute']:
            raise ValueError("method must be 'fraction' or 'absolute'")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
    
    def _compute_distance_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Efficiently compute pairwise distances between two tensors."""
        # For 1D case (after unsqueezing), use simple absolute difference
        if a.shape[1] == 1:
            return torch.abs(a - b.T)
        else:
            # For multi-dimensional case, use L2 distance by default
            # You can change this to L1 by using: torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
            return torch.cdist(a, b, p=2)
    
    def _get_kth_distances(self, distances: torch.Tensor, k: int) -> torch.Tensor:
        """Get k-th smallest distances for each row."""
        # Ensure k doesn't exceed available neighbors
        effective_k = min(k, distances.shape[1] - 1)  # -1 because we exclude self-distance
        
        if effective_k <= 0:
            # If no valid neighbors, return zeros
            return torch.zeros(distances.shape[0], device=distances.device, dtype=distances.dtype)
        
        # Use topk for efficiency when k is small relative to total points
        if effective_k <= distances.shape[1] // 4:
            return torch.topk(distances, effective_k, dim=1, largest=False, sorted=False)[0][:, -1]
        else:
            return torch.sort(distances, dim=1)[0][:, effective_k-1]
    
    def _apply_smoothing(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply smoothing kernel to distances if available."""
        if self.smoothing_kernel is None:
            return distances
        
        # Add channel dimension for conv1d
        distances_expanded = distances.unsqueeze(1)
        smoothed = F.conv1d(distances_expanded, self.smoothing_kernel, padding='same')
        return smoothed.squeeze(1)
    
    def _get_adaptive_k(self, n_points: int) -> int:
        """Get adaptive k based on available points."""
        if not self.adaptive_k:
            return self.k_neighbors
        
        # Adaptive k: use at most half of available points, but at least 1
        max_k = max(1, (n_points - 1) // 2)  # -1 to exclude self-distance
        return min(self.k_neighbors, max_k)
    
    def forward(self, points_x: torch.Tensor, points_y: torch.Tensor) -> torch.Tensor:
        """
        Compute KNN divergence loss between two point sets.
        
        Args:
            points_x (torch.Tensor): First point set [N,] or [N, D]
            points_y (torch.Tensor): Second point set [M,] or [M, D]
            
        Returns:
            torch.Tensor: KNN divergence loss
        """
        # Handle both 1D and 2D inputs
        if points_x.dim() == 1:
            points_x = points_x.unsqueeze(1)  # [N,] -> [N, 1]
        if points_y.dim() == 1:
            points_y = points_y.unsqueeze(1)  # [M,] -> [M, 1]
        
        # Validate inputs
        if points_x.dim() != 2 or points_y.dim() != 2:
            raise ValueError("Input tensors must be 1D [N,] or 2D [N, D]")
        if points_x.shape[1] != points_y.shape[1]:
            raise ValueError("Points must have same dimensionality")
        
        N, M = points_x.shape[0], points_y.shape[0]
        
        # Handle edge cases with small batch sizes
        if N < self.min_batch_size or M < self.min_batch_size:
            # Return zero loss for very small batches
            return torch.tensor(0.0, device=points_x.device, dtype=points_x.dtype)
        
        # Get adaptive k values
        k_xx = self._get_adaptive_k(N)
        k_xy = self._get_adaptive_k(M)
        
        # If adaptive k adjustment based on relative sizes
        if self.adaptive_k:
            k_multiplier = M / N
            k_xy = max(1, min(int(k_xx * k_multiplier), M - 1))
        
        # Final safety check
        if k_xx >= N or k_xy >= M:
            return torch.tensor(0.0, device=points_x.device, dtype=points_x.dtype)
        
        # Compute distance matrices
        xx_distances = self._compute_distance_matrix(points_x, points_x)
        xy_distances = self._compute_distance_matrix(points_x, points_y)
        
        # Set diagonal to infinity for xx_distances to exclude self-distances
        xx_distances.fill_diagonal_(float('inf'))
        
        # Get k-th nearest neighbor distances
        k_dist_xx = self._get_kth_distances(xx_distances, k_xx)
        k_dist_xy = self._get_kth_distances(xy_distances, k_xy)
        
        # Apply smoothing if kernel is provided
        if self.smoothing_kernel is not None:
            k_dist_xx = self._apply_smoothing(k_dist_xx.unsqueeze(1)).squeeze(1)
            k_dist_xy = self._apply_smoothing(k_dist_xy.unsqueeze(1)).squeeze(1)
        
        # Compute divergence based on method
        if self.method == "fraction":
            # Add eps for numerical stability
            output = ((1 - k_dist_xx / (k_dist_xy + self.eps)) ** 2)
        else:  # absolute
            output = (k_dist_xx - k_dist_xy) ** 2
        
        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(output)
        elif self.reduction == "sum":
            return torch.sum(output)
        else:  # none
            return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'k_neighbors={self.k_neighbors}, method={self.method}, '
                f'reduction={self.reduction}, adaptive_k={self.adaptive_k}, '
                f'min_batch_size={self.min_batch_size}, '
                f'smoothing_kernel={self.smoothing_kernel is not None}')
