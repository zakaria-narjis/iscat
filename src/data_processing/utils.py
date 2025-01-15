
import os
import nd2  
from tifffile import imwrite 
import numpy as np
from src.data_processing.labeler import Labeler
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from typing import Union, List, Tuple
import torch.nn.functional as F
class Utils:

    @staticmethod
    def get_data_paths(root_path:str, mode:str="Brightfield", image_indices:Union[List[int], int]=[0,100,200]):
        """
        Extract paths to .nd2 files and corresponding TIFF files from the specified mode folder.

        Args:
            root_path (str): The root directory to search.
            mode (str): The folder name to focus on (default is 'Brightfield').

        Returns:
            tuple: Two lists - list of .nd2 file paths and list of tuples with corresponding TIFF file paths.
        """

        zimages_files = []
        target_files = []


        for dirpath, dirnames, filenames in os.walk(root_path):
            if os.path.basename(dirpath) == mode:
                if type(image_indices) is list:
                    zimages_files.append([os.path.join(dirpath, f"frame_{idx}.tiff") for idx in image_indices])
                elif type(image_indices) is int:
                    zimages_files.append([os.path.join(dirpath, f"frame_{idx}_{image_indices}_mean.tiff") for idx in range(image_indices)])
                else:
                    raise ValueError("Invalid image_indices type")
                if  not all(os.path.exists(path) for path in zimages_files[-1]): 
                    nd2_path = None
                    for file in filenames:
                        if file.endswith('.nd2'): 
                            nd2_path = os.path.join(dirpath, file)
                            Utils.process_nd2_file(nd2_path,dirpath,image_indices)
                            break
                    if nd2_path is None:
                        raise Exception("No .nd2 file found")
                # Generate TIFF file paths dynamically based on the prefix
                cy5_path = os.path.join(dirpath, f'Captured Cy5.tif')
                fitc_path = os.path.join(dirpath, f'Captured FITC.tif')
                tritc_path = os.path.join(dirpath, f'Captured TRITC.tif')
                target_files.append((cy5_path, fitc_path, tritc_path))
                
                # Ensure all three TIFF files exist
                assert all(os.path.exists(path) for path in [cy5_path, fitc_path, tritc_path])
                assert all(os.path.exists(path) for path in zimages_files[-1])                 
                
        return zimages_files, target_files
    
    @staticmethod
    def process_nd2_file(nd2_file:str,dirpath:str,indices:Union[List[int], int])->None:
        """
        Extracts the specified frames from the .nd2 file and saves them as 16-bit TIFF files.

        """
      
        with nd2.ND2File(nd2_file) as f:
            array = f.asarray()
            if type(indices) is list:
                # Load the entire stack as a NumPy array
                num_frames = array.shape[0]  # Number of frames
                # Save each specified frame as a 16-bit TIFF
                for idx in indices:
                    assert 0 <= idx < num_frames
                    frame = array[idx]
                    # Generate the output path
                    output_path = os.path.join(
                        dirpath, 
                        f"frame_{idx}.tiff"
                    )
                    # Save the frame as a 16-bit TIFF
                    imwrite(output_path, frame, dtype='uint16')
                    print(f"Saved: {output_path}")
            elif type(indices) is int:
                frames = Utils.compute_mean_slices(array, indices)
                for idx, frame in enumerate(frames):
                    output_path = os.path.join(
                        dirpath, 
                        f"frame_{idx}_{indices}_mean.tiff"
                    )
                    imwrite(output_path, frame, dtype='uint16')
                    print(f"Saved: {output_path}")
            else:
                raise ValueError("Invalid indices type")

    @staticmethod
    def compute_mean_slices(z_stack, num_chunks):
        """
        Divides the z_stack into `num_chunks` parts along the z-axis and computes the mean 
        image for each part.

        Parameters:
            z_stack (numpy.ndarray): The input 3D numpy array with shape (200, X, X).
            num_chunks (int): Number of chunks to divide the z-axis into.

        Returns:
            list: A list of numpy arrays, each representing the mean image of a chunk.
        """
        # Ensure the z_stack is a 3D array
        if z_stack.ndim != 3:
            raise ValueError("Input z_stack must be a 3D numpy array.")

        # Get the number of slices along the z-axis
        num_slices = z_stack.shape[0]

        # Compute the size of each chunk
        chunk_size = num_slices // num_chunks

        # Initialize a list to store the mean images
        mean_images = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            # For the last chunk, include all remaining slices
            end_idx = (i + 1) * chunk_size if i != num_chunks - 1 else num_slices
            
            # Compute the mean along the z-axis for the current chunk
            mean_image = np.mean(z_stack[start_idx:end_idx], axis=0)
            mean_images.append(mean_image.astype(np.uint16))

        return mean_images
    @staticmethod
    def load_np_masks(target_paths:List[Tuple[str, str, str]],fluo_masks_indices,seg_method:str="comdet"):
        """
        Load the masks corresponding to the fluorescence images.
        """
        all_masks = []
        all_masks_paths = []
        if seg_method == "comdet":
            mask_suffix = "_mask.npy"
        elif seg_method == "kmeans":
            mask_suffix = "_mask_kmeans.npy"
        else:
            raise ValueError(f"Invalid segmentation method: {seg_method}")
        for target_path in target_paths:
            masks_path = []
            for fluo_mask_idx in fluo_masks_indices:
                mask_path =target_path[fluo_mask_idx].replace(".tif", mask_suffix)
                if os.path.exists(mask_path):
                    masks_path.append(mask_path)
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
            all_masks_paths.append(masks_path)
        for masks_path in all_masks_paths:
            masks = []
            for mask_path in masks_path:
                mask = np.load(mask_path)
                masks.append(mask)
            combined_mask = np.zeros_like(masks[0])
            for class_index, mask in enumerate(masks, start=1):
                combined_mask[mask == 1] = class_index
            # mask= sum(masks)
            # mask[mask>1]=1
            # all_masks.append(mask)
            all_masks.append(combined_mask)
        all_masks = np.concatenate([all_masks],axis=0)
        return all_masks
    
    @staticmethod
    def generate_np_masks(all_fluo_images_paths,seg_args=None,seg_method='comdet'):
        if seg_method == "comdet":
            labeler = Labeler(method="comdet")
            if seg_args is None:
                args={
                "ch1i":True,
                "ch1a":4,
                "ch1s":10
                }
            seg_args = [[args,args,args]]* len(all_fluo_images_paths)    
            for fluorescence_images_path,seg_arg in tqdm(zip(all_fluo_images_paths,seg_args),total=len(all_fluo_images_paths),desc="Creating Masks with ComDet"):
                labeler.generate_labels(fluorescence_images_path, seg_arg, segmentation_method=seg_method)
        elif seg_method == "kmeans":
                labeler = Labeler(method="kmeans")
                for fluorescence_images_path in tqdm(all_fluo_images_paths,total=len(all_fluo_images_paths),desc="Creating Masks with KMeans"):
                    labeler.generate_labels(fluorescence_images_path, segmentation_method=seg_method)
        else:
            raise ValueError("Invalid segmentation method")
        
    @staticmethod
    def visualize_labeled_canvas(mask:np.ndarray): 
        # Plot the labeled canvas
        plt.figure(figsize=(10, 10))
        plt.title("Labeled Canvas")
        plt.imshow(mask, cmap='viridis')
        plt.colorbar(label="Labels (0: Background, 1: Particle)")
        plt.axis('off')
        plt.show()
    
    # @staticmethod
    # def calculate_class_weights_from_masks(masks: torch.Tensor) -> torch.Tensor:
    #     """
    #     Calculate class weights for a binary segmentation task based on the provided masks.
    #     """
    #     class_counts = torch.zeros(2, dtype=torch.float)

    #     flattened_masks = masks.view(-1)  # Combine N, H, W into a single dimension

    #     unique, counts = torch.unique(flattened_masks.cpu(), return_counts=True)

    #     for label, count in zip(unique, counts):
    #         label = int(label)  # Ensure label is an integer
    #         class_counts[label] += count

    #     total_pixels = class_counts.sum()
    #     class_weights = total_pixels / (2 * class_counts)  # 2 is number of classes

    #     class_weights = class_weights / class_weights.sum()
        
    #     return class_weights
    
    @staticmethod
    def calculate_class_weights_from_masks(masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate class weights for a segmentation task with any number of classes.
        
        Args:
            masks (torch.Tensor): Tensor of shape (N, H, W), where N is the number of samples,
                                and each value in the masks represents a class (0 to C-1).
        
        Returns:
            torch.Tensor: Class weights of shape (C,), where C is the number of classes.
        """
        num_classes = int(masks.max().item() + 1)  # Assume classes are from 0 to C-1
        class_counts = torch.zeros(num_classes, dtype=torch.float, device=masks.device)
        flattened_masks = masks.reshape(-1)
        unique, counts = torch.unique(flattened_masks, return_counts=True)     
        for label, count in zip(unique, counts):
            class_counts[int(label)] += count  # Accumulate pixel counts for each class
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts)
        class_weights = class_weights / class_weights.sum()    
        return class_weights
    
    @staticmethod
    def z_score_normalize(images:torch.Tensor,mean:torch.Tensor,std:torch.Tensor, eps: float = 1e-8):
        normalized_images = (images - mean) / (std + eps)
        return normalized_images

    @staticmethod
    def shift_segmentation_masks(masks, shift_x=1, shift_y=1):
        """
        Shift segmentation masks in a batch to the bottom-right by specified pixels.

        Args:
            masks (torch.Tensor): A tensor of shape (B, H, W) or (B, C, H, W) representing a batch of segmentation masks.
            shift_x (int): Number of pixels to shift to the right.
            shift_y (int): Number of pixels to shift down.

        Returns:
            torch.Tensor: The shifted masks tensor with the same shape as input.
        """
        # Ensure shift_x and shift_y are non-negative
        if shift_x < 0 or shift_y < 0:
            raise ValueError("shift_x and shift_y must be non-negative.")

        # Determine padding amounts
        pad_left = shift_x
        pad_right = 0
        pad_top = shift_y
        pad_bottom = 0

        # Check if the input has a channel dimension (B, C, H, W)
        has_channels = masks.ndim == 4

        if has_channels:
            # Apply padding and slicing while keeping the batch and channel dimensions
            padded_masks = F.pad(masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            shifted_masks = padded_masks[:, :, :-shift_y if shift_y > 0 else None, :-shift_x if shift_x > 0 else None]
        else:
            # Apply padding and slicing for (B, H, W) shape
            padded_masks = F.pad(masks, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            shifted_masks = padded_masks[:, :-shift_y if shift_y > 0 else None, :-shift_x if shift_x > 0 else None]

        return shifted_masks