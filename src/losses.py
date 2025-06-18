import torch

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

    