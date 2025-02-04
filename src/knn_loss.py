import torch 

def distance_matrix(a, b):
    a_expanded = a.view(-1, 1)
    b_expanded = b.view(1, -1)

    return torch.abs(a_expanded - b_expanded)

def knn_divergence(points_x, points_y, k, smoothing_kernel=None):
    xx_distances = distance_matrix(points_x, points_x)
    xy_distances = distance_matrix(points_x, points_y) # one row for every sample in x, one col for every sample in y

    # if the sets have different sizes
    # e.g. y has twice as many points -> the distance to the 3rd closest point in x should be the same as the distance to the 6th point in y
    k_multiplier = points_y.shape[0] / points_x.shape[0]

    k_dist_xx = torch.sort(xx_distances, dim=1)[0][:, k]
    k_dist_xy = torch.sort(xy_distances, dim=1)[0][:, (k * k_multiplier).to(torch.int)]

    # optional: smoothen the distances 
    # (so that it matters less whether a point is the i-th or the (i+1)-th closest neighbor)
    if smoothing_kernel != None:
            # torch conv1d demands a channel dimension, hence the (un)squeezing
            k_dist_xx = torch.nn.functional.conv1d(k_dist_xx.unsqueeze(1), weight=smoothing_kernel.view(1, 1, -1)).flatten(1)
            k_dist_xy = torch.nn.functional.conv1d(k_dist_xy.unsqueeze(1), weight=smoothing_kernel.view(1, 1, -1)).flatten(1)

    return torch.mean((1 - k_dist_xx / k_dist_xy)**2)

    # trains more easily, but not scale-invariant. Can be useful as a first step.
    # return torch.mean((k_dist_xx - k_dist_xy)**2)

if __name__ == "__main__":
    num_points = 10000
    k = torch.arange(2, num_points/10, dtype=torch.int) # ignore the very close neighbors (k<2), they (can) make the loss too noisy
    smoothing_kernel = torch.Tensor([0.2741, 0.4519, 0.2741])

    example_points_a = torch.randn(num_points)
    example_points_b = torch.randn(num_points * 4)
    example_points_c = torch.randn(num_points) + 1
    example_points_d = torch.rand(num_points)

    print(knn_divergence(example_points_a, example_points_b, k))
    print(knn_divergence(example_points_a, example_points_c, k))
    print(knn_divergence(example_points_a, example_points_d, k))

    

