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

class iScatDataset(Dataset):
    def __init__(self, image_paths, target_paths, seg_args=None,image_size=(224,224),train=True,preload_image=False,reload_mask=False,apply_augmentation=True,duplication_factor=100,normalize=True,seg_method="comdet",fluo_masks_indices=[0,1,2],device="cpu"):
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
        if self.preload_image:
            self.images = []
            for z_images in tqdm(self.image_paths, desc="Loading TIFF images to Memory"):
                # Read each frame and stack them into a single array
                frames = [imread(frame_path) for frame_path in z_images]
                stacked_frames = np.stack(frames, axis=0)  # Shape: (3, H, W)
                self.images.append(stacked_frames)
            self.images = np.concatenate([self.images],axis=0)
            self.images = torch.from_numpy(self.images)
            self.images=self.images.to(dtype=torch.float32)
            if self.normalize:
                self.images = self.normalize_image(self.images)
        self.image_paths = np.concatenate([self.image_paths])
        self.image_paths = np.repeat(self.image_paths,self.duplication_factor,axis=0)

        self.masks =  Utils.load_np_masks(target_paths,self.fluo_masks_indices,seg_method=self.seg_method)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(dtype=torch.float32)

        self.masks = self.masks.to(self.device)
        self.images = self.images.to(self.device)


    def normalize_image(self, image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], eps: float = 1e-8):
        """
        Normalize an image: cast to float32 and normalize using mean and std.
        Args:
            image (torch.Tensor): Input image.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
        Returns:
            torch.Tensor: Normalized image in float32 format.
        """
        normalized_images = image/65535.0
        # Compute mean and std for each image in the batch
        mean = normalized_images .mean(dim=( 2, 3), keepdim=True)  # Shape: (N, 1, 1, 1)
        std = normalized_images .std(dim=( 2, 3), keepdim=True)    # Shape: (N, 1, 1, 1)
        
        # Perform z-score normalization
        normalized_images = (normalized_images - mean) / (std + eps)
        return normalized_images

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
            image = nd2.imread(self.image_paths[index])
        mask = self.masks[index_in_images]
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)