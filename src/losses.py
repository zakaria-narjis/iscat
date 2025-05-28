# import torch

# def distance_matrix(a, b):
#     a_expanded = a.view(-1, 1)
#     b_expanded = b.view(1, -1)

#     return torch.abs(a_expanded - b_expanded)

# def knn_divergence(points_x, points_y, k, smoothing_kernel=None):
#     xx_distances = distance_matrix(points_x, points_x)
#     xy_distances = distance_matrix(points_x, points_y) # one row for every sample in x, one col for every sample in y

#     # if the sets have different sizes
#     # e.g. y has twice as many points -> the distance to the 3rd closest point in x should be the same as the distance to the 6th point in y
#     k_multiplier = points_y.shape[0] / points_x.shape[0]

#     k_dist_xx = torch.sort(xx_distances, dim=1)[0][:, k]
#     k_dist_xy = torch.sort(xy_distances, dim=1)[0][:, (k * k_multiplier).to(torch.int)]

#     # optional: smoothen the distances
#     # (so that it matters less whether a point is the i-th or the (i+1)-th closest neighbor)
#     if smoothing_kernel != None:
#             # torch conv1d demands a channel dimension, hence the (un)squeezing
#             k_dist_xx = torch.nn.functional.conv1d(k_dist_xx.unsqueeze(1), weight=smoothing_kernel.view(1, 1, -1)).flatten(1)
#             k_dist_xy = torch.nn.functional.conv1d(k_dist_xy.unsqueeze(1), weight=smoothing_kernel.view(1, 1, -1)).flatten(1)

#     return torch.mean((1 - k_dist_xx / k_dist_xy)**2)

#     # trains more easily, but not scale-invariant. Can be useful as a first step.
#     # return torch.mean((k_dist_xx - k_dist_xy)**2)

# if __name__ == "__main__":
#     num_points = 10000
#     k = torch.arange(2, num_points/10, dtype=torch.int) # ignore the very close neighbors (k<2), they (can) make the loss too noisy
#     smoothing_kernel = torch.Tensor([0.2741, 0.4519, 0.2741])

#     example_points_a = torch.randn(num_points)
#     example_points_b = torch.randn(num_points * 4)
#     example_points_c = torch.randn(num_points) + 1
#     example_points_d = torch.rand(num_points)

#     print(knn_divergence(example_points_a, example_points_b, k))
#     print(knn_divergence(example_points_a, example_points_c, k))
#     print(knn_divergence(example_points_a, example_points_d, k))


import torch


def distance_matrix(a, b):
    a_expanded = a.view(-1, 1)
    b_expanded = b.view(1, -1)

    return torch.abs(a_expanded - b_expanded)


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

  
def compute_contrast(imgs: torch.Tensor, dim=(1, 2, 3)):
    return imgs.amax(dim=dim) - imgs.amin(dim=dim)

def MonotonicityLoss(predictions: torch.Tensor, images: torch.Tensor, direction: str = "increasing"):
    """
    Enforces that predictions follow the same monotonic order as contrast.
    
    Args:
        predictions (torch.Tensor): Predicted values of shape (B,)
        images (torch.Tensor): Batch of images of shape (B, C, H, W)
        direction (str): 'increasing' or 'decreasing' monotonicity
        
    Returns:
        torch.Tensor: Scalar monotonicity loss
    """
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")

    contrast_values = compute_contrast(images)  # Shape: (B,)
    predictions = predictions.view(-1)          # Ensure shape (B,)

    contrast_diff = contrast_values.unsqueeze(0) - contrast_values.unsqueeze(1)  # (B, B)
    pred_diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)              # (B, B)

    if direction == "increasing":
        # Enforce: if contrast_i < contrast_j, then pred_i < pred_j
        contrast_pairs = contrast_diff < 0
        violations = torch.relu(-pred_diff) * contrast_pairs  # Penalize pred_i >= pred_j
    else:  # "decreasing"
        # Enforce: if contrast_i < contrast_j, then pred_i > pred_j
        contrast_pairs = contrast_diff < 0
        violations = torch.relu(pred_diff) * contrast_pairs   # Penalize pred_i <= pred_j
    # violations = violations**2  # Square the violations for loss calculation
    num_pairs = contrast_pairs.sum()
    loss = violations.sum() / (num_pairs + 1e-8)
    return loss
