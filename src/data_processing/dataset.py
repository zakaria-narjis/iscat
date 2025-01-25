from torchvision.transforms.v2 import functional as TF
import random
from torch.utils.data import Dataset
import numpy as np
import torch
from tifffile import imread
from src.data_processing.utils import Utils
import h5py
import torch
from torch.utils.data import Dataset
import random

class iScatDataset(Dataset):
    def __init__(self, hdf5_path, classes=[0, 1, 2], apply_augmentation=False, normalize="minmax", indices=None,multi_class=False):
        """
        PyTorch Dataset for microscopy data stored in an HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            classes (list): Classes to include in the mask.
            apply_augmentation (bool): Whether to apply random flips.
            normalize (str): Normalization method, either 'minmax' or 'zscore'.
            indices (list): Optional list of indices to subset the dataset.
        """
        self.hdf5_path = hdf5_path
        self.classes = classes
        self.apply_augmentation = apply_augmentation
        self.normalize = normalize
        self.multi_class = multi_class
        # Open HDF5 file and get dataset sizes
        with h5py.File(hdf5_path, "r") as f:
            self.image_dataset_size = f["image_patches"].shape[0]

        # If indices are provided, filter dataset length
        self.indices = indices if indices is not None else range(self.image_dataset_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the input index to the subset index
        idx = self.indices[idx]

        # Load the data from HDF5
        with h5py.File(self.hdf5_path, "r") as f:
            image = f["image_patches"][idx].copy()  # Shape: (Z, H, W)
            masks = f["mask_patches"][idx].copy()   # Shape: (C, H, W)
        image = Utils.extract_averaged_frames(image, num_frames=32)
        # Convert to float32
        image = torch.from_numpy(image.astype(np.float32))  # Convert image to tensor
        masks = torch.from_numpy(masks.astype(np.uint8))    # Convert masks to tensor

        # Normalize the image
        if self.normalize == "minmax":
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.normalize == "zscore":
            mean = image.mean()
            std = image.std() + 1e-8
            image = (image - mean) / std
        elif self.normalize is None:
            pass
        else:
            raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")

        # Process masks based on selected classes
        if len(self.classes) == 1:
            # Single-class mask
            mask = masks[self.classes[0]]
        else:
            # Multi-class mask
            if self.multi_class:
                mask = torch.zeros_like(masks[0], dtype=torch.uint8)
                for i, cls in enumerate(self.classes, start=1):
                    mask[masks[cls] > 0] = i  # Assign class indices
            else:
                # Binary mask
                mask = torch.zeros_like(masks[0], dtype=torch.uint8)
                for cls in self.classes:
                    mask += masks[cls]
                mask[mask>1] = 1  # Ensure binary mask
                
        # Apply augmentation using torchvision.transforms.functional
        if self.apply_augmentation:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation of 90° or -90°
            if random.random() > 0.5:
                angle = random.choice([90, -90])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)        
        # Ensure the returned tensors have the right shapes
        return image, mask