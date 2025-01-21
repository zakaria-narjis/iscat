from torchvision.transforms.v2 import Resize, RandomCrop, Normalize
from torchvision.transforms.v2 import functional as TF
import random
from torch.utils.data import Dataset
import nd2
import numpy as np
import torch
from tqdm import tqdm
import os
from tifffile import imread
from src.data_processing.utils import Utils
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class iScatDataset(Dataset):
    def __init__(self, image_paths, target_paths, seg_args=None,image_size=(224,224),train=True,preload_image=False,reload_mask=False,apply_augmentation=True,duplication_factor=100,normalize=True,seg_method="comdet",fluo_masks_indices=[0,1,2],device="cpu",apply_mask_correction=True):
        self.image_paths = image_paths
        self.target_paths = target_paths #list of tuple of paths
        self.seg_args = seg_args
        self.image_size = image_size
        self.preload_image = preload_image
        self.seg_args = seg_args
        self.seg_method = seg_method
        self.fluo_masks_indices = fluo_masks_indices #list of indices of the fluorescence images to use
        self.apply_augmentation = apply_augmentation
        self.duplication_factor = duplication_factor #number of times to repeat the image
        self.normalize = normalize
        self.device = device
        
        self.images = []
        for z_images in tqdm(self.image_paths, desc="Loading images to Memory"):
            # Read each frame and stack them into a single array
            frames = [imread(frame_path) for frame_path in z_images]
            stacked_frames = np.stack(frames, axis=0)  # Shape: (3, H, W)
            self.images.append(stacked_frames)
        self.images = np.concatenate([self.images],axis=0)
        self.images = torch.from_numpy(self.images)
        self.images=self.images.to(dtype=torch.float32)
        self.mean = self.images.mean(dim=(0, 2, 3), keepdim=True).to(self.device) 
        self.std = self.images.std(dim=(0, 2, 3), keepdim=True).to(self.device)
        if self.preload_image:
            self.images = self.images.to(self.device)     
        else:
             self.images = []
        self.image_paths = np.concatenate([self.image_paths])
        # self.image_paths = np.repeat(self.image_paths,self.duplication_factor,axis=0)


        self.masks =  Utils.load_np_masks(target_paths,self.fluo_masks_indices,seg_method=self.seg_method)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(dtype=torch.float32)
        self.masks = self.masks.to(self.device)
        if apply_mask_correction:
            self.masks = Utils.shift_segmentation_masks(self.masks,shift_x=2,shift_y=1)

    def augment(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask
    
    def transform(self, image, mask):
        # Random crop
        i, j, h, w = RandomCrop.get_params(
            image, output_size=self.image_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        if self.apply_augmentation:
            image,mask = self.augment(image,mask)
        # Transform to tensor
        return image, mask

    def __getitem__(self, index):
        index_in_images = index//self.duplication_factor
        if self.preload_image:
            image = self.images[index_in_images]
        else:
            image = [imread(frame_path) for frame_path in self.image_paths[index_in_images]]
            image = np.stack(image, axis=0)  # Shape: (12, H, W)
            image = torch.from_numpy(image).float()
            image = image.to(dtype=torch.float32)         
        mask = self.masks[index_in_images]
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        length = len(self.image_paths)*self.duplication_factor
        return length
class iScatDataset2(Dataset):
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