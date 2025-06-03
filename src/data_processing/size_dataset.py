import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
        #         size=(32, 64),
        #         mode="bicubic",
        #         align_corners=True,
        #     )
        #     .squeeze(0)
        #     .squeeze(0)
        # )  # Remove batch and channel dims
        resized = particle_tensor.squeeze(0)
        final_tensor = resized.unsqueeze(0).repeat(
            3, 1, 1
        )  # Repeat across 3 channels

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
