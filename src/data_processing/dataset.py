from torchvision.transforms.v2 import Resize, RandomCrop, Normalize
from torchvision.transforms.v2 import functional as TF
import random
from torch.utils.data import Dataset
import nd2
from src.data_processing.labeler import Labeler
import numpy as np
import torch
from tqdm import tqdm
import os

class iScatDataset(Dataset):
    def __init__(self, image_paths, target_paths, seg_args=None,image_size=(224,224),train=True,preload_image=False,reload_mask=False,apply_augmentation=True):
        self.image_paths = image_paths
        self.target_paths = target_paths #list of tuple of paths
        self.seg_args = seg_args
        self.image_size = image_size
        self.preload_image = preload_image
        self.seg_args = seg_args
        self.apply_augmentation = apply_augmentation
        self.duplication_factor = 100 #number of times to repeat the image
        if self.preload_image:
            self.images = []
            for image_path in tqdm(self.image_paths,desc="Loading surface images to Memory"):
                self.images.append(nd2.imread(image_path)[[1,100,199],:,:])           
            self.images = np.concatenate([self.images],axis=0)
            self.images = torch.from_numpy(self.images)
            self.images = self.normalize_image(self.images)
        self.image_paths = np.concatenate([self.image_paths])
        self.image_paths = np.repeat(self.image_paths,self.duplication_factor,axis=0)

        if reload_mask or not all([os.path.exists(os.path.join(os.path.dirname(target_path[0]),"mask.npy"))for target_path in self.target_paths]):
            self.labeler = Labeler()
        #default segmentation arguments
        if self.seg_args is None:
            args={
            "ch1i":True,
            "ch1a":4,
            "ch1s":10
            }
            self.seg_args = [[args,args,args]]*len(self.target_paths)
        self.masks= []

        for fluorescence_images_paths, seg_args in tqdm(
            zip(self.target_paths, self.seg_args),
            total=len(self.target_paths), 
            desc="Creating Masks"      
        ):
            if reload_mask==False and os.path.exists(os.path.join(os.path.dirname(fluorescence_images_paths[0]),"mask.npy")):
                mask = np.load(os.path.join(os.path.dirname(fluorescence_images_paths[0]),"mask.npy"))
            else:                   
                mask = self.labeler.label(fluorescence_images_paths, seg_args, segmentation_method="comdet")
                np.save(os.path.join(os.path.dirname(fluorescence_images_paths[0]),"mask.npy"),mask)
            self.masks.append(mask)
        self.masks = np.concatenate([self.masks],axis=0)
        self.masks = torch.from_numpy(self.masks).float()

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