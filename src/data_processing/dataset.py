from torchvision.transforms.v2 import Resize, RandomCrop
from torchvision.transforms.v2 import functional as TF
from random import random
from torch.utils.data import Dataset
import nd2
from data_processing.labeler import Labeler
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, seg_args=None,image_size=(224,224),train=True,preload_image=False):
        self.image_paths = image_paths
        self.target_paths = target_paths #list of tuple of paths
        self.seg_args = seg_args
        self.image_size = image_size
        self.preload_image = preload_image
        self.seg_args = seg_args
        if self.preload_image:
            self.images = np.concatenate([nd2.imread(image_path)[0] for image_path in self.image_paths])
            self.images = torch.from_numpy(self.images)
        else:
            self.image_paths = self.image_paths*100
            self.target_paths = self.target_paths
            self.image_paths = np.concatenate(self.image_paths)
        self.Labeller = Labeler()

        #default segmentation arguments
        if self.seg_args is None:
            args={
            "ch1i":True,
            "ch1a":4,
            "ch1s":10
            }
            self.args = [[args,args,args]]
        self.masks= []
        for fluorescence_images_paths,seg_args in zip(self.target_paths,self.seg_args):
            mask = self.labeler.label(fluorescence_images_paths,seg_args,segmentation_method="comdet")
            self.masks.append(mask)
        self.masks = np.concatenate(self.masks,axis=0)
        self.masks = np.repeat(self.masks,100,axis=0)
        self.masks = torch.from_numpy(self.masks).float()

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
        image,mask = self.augment(image,mask)
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        if self.preload_image:
            image = self.images[index]
        else:
            image = nd2.imread(self.image_paths[index])
        mask = self.masks[index]
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)