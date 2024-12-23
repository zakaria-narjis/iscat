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
        self.masks = self.one_hot_mask(self.masks)
        self.masks = self.masks.to(dtype=torch.float32)

        self.masks = self.masks.to(self.device)
        self.images = self.images.to(self.device)

    def one_hot_mask(self, mask):
        """
        Convert a single-channel mask with class indices into a one-hot encoded mask with multiple channels.
        
        Args:
            mask (torch.Tensor): Single-channel mask of shape (H, W) with values corresponding to class indices.
            
        Returns:
            torch.Tensor: One-hot encoded mask of shape (num_classes, H, W).
        """
        num_classes = 3  # Adjust if you have more classes
        # Ensure the mask is of type long (necessary for one-hot encoding)
        mask = mask.long()
        # Create one-hot encoded mask
        one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=num_classes)
        # Rearrange dimensions to (num_classes, H, W)
        one_hot_mask = one_hot_mask.permute(0, 3, 1, 2)
        return one_hot_mask

    def normalize_image(self, image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        """
        Normalize an image: cast to float32 and normalize using mean and std.
        Args:
            image (torch.Tensor): Input image.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
        Returns:
            torch.Tensor: Normalized image in float32 format.
        """
        image = image.to(dtype=torch.float32)       
        # image = image / image.amax(dim=(2,3), keepdim=True)
        # mean = torch.tensor(mean, dtype=torch.float32, device=image.device).view(1, -1, 1, 1)
        # std = torch.tensor(std, dtype=torch.float32, device=image.device).view(1, -1, 1, 1)
        # out = (image - mean) / std
        # image = image / (2**16-1)
        image = (image-237)/(15321-237)
        out = Normalize(mean,std)(image)
        return out

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