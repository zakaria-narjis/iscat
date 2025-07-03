import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import warnings

def compute_normalization_stats(h5_path, classes=None):
    """
    Compute mean and standard deviation for z-score normalization.

    Args:
        h5_path (str): Path to HDF5 file
        classes (list, optional): List of classes to include in computation

    Returns:
        tuple: (mean, std) computed across all data points
    """
    with h5py.File(h5_path, "r") as h5_file:
        data = h5_file["data"][:]
        labels = h5_file["labels"][:]

        if classes is not None:
            # Filter data for selected classes
            mask = np.isin(labels, classes)
            data = data[mask]

        # Compute statistics across all dimensions
        mean = np.mean(data)
        std = np.std(data)

        print(f"Computed statistics: mean = {mean:.4f}, std = {std:.4f}")

        return mean, std


class ParticleDataset(Dataset):
    """Custom Dataset for particle data with flexible class selection and normalization."""

    def __init__(
        self,
        h5_path,
        classes=[0, 1],
        transform=None,
        mean=None,
        std=None,
        padding=False,
        indices=None,
    ):
        # self.h5_file = h5py.File(h5_path, 'r')
        # data = self.h5_file['data'][:]
        # labels = self.h5_file['labels'][:]
        self.padding = padding
        # Filter data for selected classes

        with h5py.File(h5_path, "r") as h5_file:
            mask = np.isin(h5_file["labels"], classes)
            if indices is None:
                self.data = h5_file["data"][mask][:]
                self.labels = h5_file["labels"][mask][:]
            else:
                self.data = h5_file["data"][mask][indices]
                self.labels = h5_file["labels"][mask][indices]

        # Create class mapping to handle non-consecutive class indices
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

        # Map original labels to new consecutive indices
        self.labels = np.array(
            [self.class_to_idx[label] for label in self.labels]
        )
        self.transform = transform
        if mean is None or std is None:
            self.mean, self.std = compute_normalization_stats(h5_path, classes)
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get particle data
        particle = self.data[idx]  # Shape: (16, 201)

        # Apply normalization if mean and std are provided
        if self.mean is not None and self.std is not None:
            particle = (particle - self.mean) / self.std

        # Convert to torch tensor for better interpolation
        particle_tensor = torch.FloatTensor(particle).unsqueeze(
            0
        )  # Add channel dim

        # Resize to (16, 16) using bicubic interpolation
        # resized = (
        #     torch.nn.functional.interpolate(
        #         particle_tensor.unsqueeze(0),  # Add batch dim
        #         size=(16, 128),
        #         mode="bicubic",
        #         align_corners=True,
        #     )
        #     .squeeze(0)
        #     .squeeze(0)
        # Remove batch and channel dims
        resized = particle_tensor.squeeze(0)
        # final_tensor = resized.unsqueeze(0).repeat(
        #     3, 1, 1
        # )  # Repeat across 3 channels
        final_tensor = resized.unsqueeze(0).repeat(
            1, 1, 1
        )  
        if self.transform:
            final_tensor = self.transform(final_tensor)

        # Create one-hot encoded label
        label_idx = self.labels[idx]
        # label_onehot = torch.zeros(self.num_classes)
        # label_onehot[label_idx] = 1

        # return final_tensor, label_onehot
        return final_tensor, label_idx

    def close(self):
        self.h5_file.close()

def contrast_from_middle_rows(images: np.array,offset=2) -> torch.Tensor:
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
    contrast = gray.std(axis=(1, 2))  # Compute std over height and width

    return contrast

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    ## type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def assign_p_by_contrast(data, Cls, sizes):
    C = contrast_from_middle_rows(data)  # (N,)
    N = len(Cls)
    P_assigned = np.empty(N, dtype=np.float32)

    for class_id, P in enumerate(sizes):
        idx = np.where(Cls == class_id)[0]
        sorted_idx = idx[np.argsort(C[idx])]
        sorted_P = np.sort(P)
        P_assigned[sorted_idx] = sorted_P

    return P_assigned


class ParticleDatasetReg(Dataset):

    def __init__(
        self,
        h5_path,
        classes=[0, 1],
        transform=None,
        mean=None,
        std=None,
        padding=False,
        indices=None,
    ):
        self.padding = padding
        # Filter data for selected classes

        with h5py.File(h5_path, "r") as h5_file:
            mask = np.isin(h5_file["labels"], classes)
            if indices is None:
                self.data = h5_file["data"][mask][:]
                self.labels = h5_file["labels"][mask][:]
            else:
                self.data = h5_file["data"][mask][indices]
                self.labels = h5_file["labels"][mask][indices]

        # Create class mapping to handle non-consecutive class indices
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)

        # Map original labels to new consecutive indices
        self.labels = np.array(
            [self.class_to_idx[label] for label in self.labels]
        )
        self.transform = transform
        if mean is None or std is None:
            self.mean, self.std = compute_normalization_stats(h5_path, classes)
        else:
            self.mean = mean
            self.std = std
        particles_stats = {0:(80, 22.5,10,float("inf")),
                           1:(302, 25,-float("inf"),float("inf")),
                           2:(626, 128,-float("inf"),float("inf")),
                           3:(1300, 150,-float("inf"),float("inf"))}
        particles_stats = {self.class_to_idx[k]: v for k, v in particles_stats.items() if k in classes}
        distributions = [
            trunc_normal_(
                torch.empty(len(self.labels[self.labels==k])),v[0], v[1], v[2], v[3]
            ) for k, v in particles_stats.items()
        ]
        self.size_labels = assign_p_by_contrast(
            self.data[:,np.newaxis,...], self.labels, distributions
        )
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get particle data
        particle = self.data[idx]  # Shape: (16, 201)

        # Apply normalization if mean and std are provided
        if self.mean is not None and self.std is not None:
            particle = (particle - self.mean) / self.std

        # Convert to torch tensor for better interpolation
        particle = torch.from_numpy(particle).unsqueeze(
            0
        ).squeeze(0).repeat(
            1, 1, 1
        ).to(torch.float32)
        if self.transform:
            particle = self.transform(particle)

        # Create one-hot encoded label
        class_label = self.labels[idx]
        size_label = self.size_labels[idx]
        return particle, class_label, size_label

    def close(self):
        self.h5_file.close()