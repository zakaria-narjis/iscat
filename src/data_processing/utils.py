
import os
import nd2  
from tifffile import imwrite 
import numpy as np
from src.data_processing.labeler import Labeler
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
class Utils_legacy:
    @staticmethod
    def get_data_paths(root_path, mode="Brightfield"):
        """
        Extract paths to .nd2 files and corresponding TIFF files from the specified mode folder.

        Args:
            root_path (str): The root directory to search.
            mode (str): The folder name to focus on (default is 'Brightfield').

        Returns:
            tuple: Two lists - list of .nd2 file paths and list of tuples with corresponding TIFF file paths.
        """
        nd2_files = []
        tiff_files = []
        
        for dirpath, dirnames, filenames in os.walk(root_path):
            if os.path.basename(dirpath) == mode:  # Focus on the specified mode folder
                for file in filenames:
                    if file.endswith('.nd2'):  # Check for .nd2 files
                        nd2_path = os.path.join(dirpath, file)
                        
                        # Generate TIFF file paths dynamically based on the prefix
                        cy5_path = os.path.join(dirpath, f'Captured Cy5.tif')
                        fitc_path = os.path.join(dirpath, f'Captured FITC.tif')
                        tritc_path = os.path.join(dirpath, f'Captured TRITC.tif')
                        
                        # Ensure all three TIFF files exist
                        if all(os.path.exists(path) for path in [cy5_path, fitc_path, tritc_path]):
                            nd2_files.append(nd2_path)
                            tiff_files.append((cy5_path, fitc_path, tritc_path))
        
        return nd2_files, tiff_files
    
    @staticmethod
    def process_nd2_file(file_path,indices):
    # Open the .nd2 file
        with nd2.ND2File(file_path) as f:
            array = f.asarray()  # Load the entire stack as a NumPy array
            num_frames = array.shape[0]  # Number of frames
            # Save each specified frame as a 16-bit TIFF
            for idx in indices:
                if 0 <= idx < num_frames:
                    frame = array[idx]
                    # Generate the output path
                    output_path = os.path.join(
                        os.path.dirname(file_path), 
                        f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{idx:03d}.tiff"
                    )
                    # Save the frame as a 16-bit TIFF
                    imwrite(output_path, frame, dtype='uint16')
                    print(f"Saved: {output_path}")

class Utils:
    @staticmethod
    def get_data_paths(root_path, mode="Brightfield", image_indices=[0,100,200]):
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
                 
                zimages_files.append([os.path.join(dirpath, f"frame_{idx}.tiff") for idx in image_indices])
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
    def process_nd2_file(nd2_file,dirpath,indices):
    # Open the .nd2 file
        with nd2.ND2File(nd2_file) as f:
            array = f.asarray()  # Load the entire stack as a NumPy array
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

    @staticmethod
    def load_np_masks(target_paths,fluo_masks_indices,seg_method="comdet"):
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
            mask= sum(masks)
            mask[mask>1]=1
            all_masks.append(mask)
        all_masks = np.concatenate([all_masks],axis=0)
        return all_masks
    
    @staticmethod
    def generate_np_masks(all_fluo_images_paths,seg_args=None,seg_method="comdet"):
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
    def visualize_labeled_canvas(mask): 
        # Plot the labeled canvas
        plt.figure(figsize=(10, 10))
        plt.title("Labeled Canvas")
        plt.imshow(mask, cmap='viridis')
        plt.colorbar(label="Labels (0: Background, 1: Particle)")
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def calculate_class_weights_from_masks(masks):

        class_counts = torch.zeros(2, dtype=torch.float)

        flattened_masks = masks.view(-1)  # Combine N, H, W into a single dimension

        unique, counts = torch.unique(flattened_masks.cpu(), return_counts=True)

        for label, count in zip(unique, counts):
            label = int(label)  # Ensure label is an integer
            class_counts[label] += count

        total_pixels = class_counts.sum()
        class_weights = total_pixels / (2 * class_counts)  # 2 is number of classes

        class_weights = class_weights / class_weights.sum()
        
        return class_weights
    
    @staticmethod
    def z_score_normalize(images,mean,std, eps: float = 1e-8):
        normalized_images = (images - mean) / (std + eps)
        return normalized_images