import os
import h5py
import numpy as np
import pandas as pd
from nd2 import ND2File
import argparse
from typing import Dict, List, Tuple

def get_nd2_paths(base_path, option):
    """
    Recursively collects paths to .nd2 files inside specified subfolders of Metasurface directories.

    Args:
        base_path (str): The base directory to search.
        option (str): The folder to consider ('Brightfield' or 'Laser').

    Returns:
        list: A list of paths to .nd2 files.
    """
    if option not in {'Brightfield', 'Laser'}:
        raise ValueError("Option must be 'Brightfield' or 'Laser'")
    
    nd2_paths = []
    
    for root, dirs, files in os.walk(base_path):
        if 'Metasurface' in os.path.basename(root):
            target_folder = os.path.join(root, option)
            if os.path.isdir(target_folder):
                for file in os.listdir(target_folder):
                    if file.endswith('.nd2'):
                        nd2_paths.append(os.path.join(target_folder, file))  
    return nd2_paths
class InstanceMaskGenerator:
    def __init__(self):
        self.class_info = {
            "Captured Cy5": {"id": 0, "name": "Cy5", "size": "80nm"},
            "Captured FITC": {"id": 1, "name": "FITC", "size": "300nm"},
            "Captured TRITC": {"id": 2, "name": "TRITC", "size": "1300nm"}
        }
        
        # Special case handling for 2024_11_29
        self.rename_mapping = {
            "Captured Cy5": "Captured FITC",
            "Captured FITC": "Captured TRITC",
            "Captured TRITC": "Captured Cy5"
        }

    def create_instance_mask(self, canvas_shape: Tuple[int, int], 
                           particles_df: pd.DataFrame) -> np.ndarray:
        """
        Create instance mask where each instance has a unique ID
        """
        instance_canvas = np.zeros(canvas_shape, dtype=np.uint16)
        y, x = np.ogrid[:canvas_shape[0], :canvas_shape[1]]
        
        for idx, row in particles_df.iterrows():
            instance_id = idx + 1  # Start from 1, 0 is background
            center_x = int((row['xMin'] + row['xMax']) / 2)
            center_y = int((row['yMin'] + row['yMax']) / 2)
            axes_x = max(int((row['xMax'] - row['xMin']) / 2), 1)
            axes_y = max(int((row['yMax'] - row['yMin']) / 2), 1)
            
            ellipse_mask = ((x - center_x) / axes_x)**2 + ((y - center_y) / axes_y)**2 <= 1
            instance_canvas[ellipse_mask] = instance_id
            
        return instance_canvas

    def extract_patch_instances(self, instance_mask: np.ndarray, 
                              y_start: int, x_start: int, 
                              patch_size: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        """
        Extract patch from instance mask and remap instance IDs to be consecutive
        """
        patch = instance_mask[y_start:y_start + patch_size[0], 
                            x_start:x_start + patch_size[1]].copy()
        
        # Remap instance IDs to be consecutive starting from 1
        unique_ids = np.unique(patch)
        unique_ids = unique_ids[unique_ids != 0]  # Exclude background
        
        id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
        id_map[0] = 0  # Keep background as 0
        
        for old_id, new_id in id_map.items():
            patch[patch == old_id] = new_id
            
        return patch, len(unique_ids)

def process_data(nd2_paths: List[str], 
                output_path: str,
                selected_classes: List[int],
                patch_size: Tuple[int, int] = (256, 256),
                overlap: int = 0,
                mode: str = "instance"):
    """
    Process ND2 files and create HDF5 dataset for instance segmentation
    """
    mask_gen = InstanceMaskGenerator()
    patch_height, patch_width = patch_size
    
    with h5py.File(output_path, 'w') as hdf5_file:
        image_dataset = None
        instance_dataset = None
        semantic_dataset = None
        bbox_dataset = None
        
        # Store metadata
        hdf5_file.attrs["description"] = "Instance segmentation dataset"
        hdf5_file.attrs["selected_classes"] = selected_classes
        
        for nd2_path in nd2_paths:
            print(f"Processing {nd2_path}...")
            is_special_case = "2024_11_29" in nd2_path
            
            # Load image
            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()
                num_slices, height, width = image.shape
            
            # Process each class
            full_instance_masks = []
            for class_name, info in mask_gen.class_info.items():
                if info["id"] not in selected_classes:
                    continue
                    
                # Handle special case renaming
                csv_name = f"{class_name}.csv"
                if is_special_case:
                    csv_name = f"{mask_gen.rename_mapping[class_name]}.csv"
                
                csv_path = os.path.join(os.path.dirname(nd2_path), csv_name)
                if not os.path.exists(csv_path):
                    print(f"Warning: {csv_path} not found, skipping...")
                    continue
                
                # Load and process particle data
                particles_df = pd.read_csv(csv_path)
                instance_mask = mask_gen.create_instance_mask((height, width), particles_df)
                full_instance_masks.append((instance_mask, info["id"]))
            
            # Extract patches
            for y in range(0, height - patch_height + 1, patch_height - overlap):
                for x in range(0, width - patch_width + 1, patch_width - overlap):
                    # Extract image patch
                    image_patch = image[:, y:y + patch_height, x:x + patch_width]
                    
                    # Process instance masks for this patch
                    instance_patches = []
                    semantic_patches = []
                    bbox_info = []
                    
                    for instance_mask, class_id in full_instance_masks:
                        patch_mask, num_instances = mask_gen.extract_patch_instances(
                            instance_mask, y, x, patch_size)
                        
                        if num_instances > 0:
                            instance_patches.append(patch_mask)
                            
                            # Create semantic mask (binary mask for each class)
                            semantic_mask = (patch_mask > 0).astype(np.uint8)
                            semantic_patches.append(semantic_mask)
                            
                            # Store bounding box information
                            for instance_id in range(1, num_instances + 1):
                                instance_pixels = np.where(patch_mask == instance_id)
                                if len(instance_pixels[0]) > 0:
                                    ymin, xmin = np.min(instance_pixels, axis=1)
                                    ymax, xmax = np.max(instance_pixels, axis=1)
                                    bbox_info.append([class_id, instance_id, xmin, ymin, xmax, ymax])
                    
                    if not instance_patches:
                        continue
                    
                    # Initialize or resize datasets
                    if image_dataset is None:
                        image_dataset = hdf5_file.create_dataset(
                            "images",
                            shape=(0, num_slices, patch_height, patch_width),
                            maxshape=(None, num_slices, patch_height, patch_width),
                            chunks=(1, num_slices, patch_height, patch_width),
                            dtype=image_patch.dtype
                        )
                        
                        instance_dataset = hdf5_file.create_dataset(
                            "instance_masks",
                            shape=(0, len(selected_classes), patch_height, patch_width),
                            maxshape=(None, len(selected_classes), patch_height, patch_width),
                            chunks=(1, len(selected_classes), patch_height, patch_width),
                            dtype=np.uint16
                        )
                        
                        semantic_dataset = hdf5_file.create_dataset(
                            "semantic_masks",
                            shape=(0, len(selected_classes), patch_height, patch_width),
                            maxshape=(None, len(selected_classes), patch_height, patch_width),
                            chunks=(1, len(selected_classes), patch_height, patch_width),
                            dtype=np.uint8
                        )
                        
                        bbox_dataset = hdf5_file.create_dataset(
                            "bboxes",
                            shape=(0, 6),  # [class_id, instance_id, xmin, ymin, xmax, ymax]
                            maxshape=(None, 6),
                            chunks=(1, 6),
                            dtype=np.float32
                        )
                    
                    # Save patches
                    image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                    image_dataset[-1] = image_patch
                    
                    instance_dataset.resize(instance_dataset.shape[0] + 1, axis=0)
                    instance_dataset[-1] = np.stack(instance_patches)
                    
                    semantic_dataset.resize(semantic_dataset.shape[0] + 1, axis=0)
                    semantic_dataset[-1] = np.stack(semantic_patches)
                    
                    if bbox_info:
                        current_size = bbox_dataset.shape[0]
                        bbox_dataset.resize(current_size + len(bbox_info), axis=0)
                        bbox_dataset[current_size:] = bbox_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 dataset for instance segmentation")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], required=True)
    parser.add_argument("--output_path", type=str, default='dataset')
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--classes", type=int, nargs='+', default=[0, 1, 2],
                      help="Class IDs to include (0: Cy5, 1: FITC, 2: TRITC)")
    args = parser.parse_args()

    # Setup paths
    data_paths = [
        os.path.join('data', '2024_11_11', 'Metasurface', 'Chip_02'),
        os.path.join('data', '2024_11_12', 'Metasurface', 'Chip_01'),
        # os.path.join('data', '2024_11_29', 'Metasurface', 'Chip_02')
    ]
    
    nd2_paths = []
    for data_path in data_paths:
        nd2_paths.extend(get_nd2_paths(data_path, args.datatype))
    
    os.makedirs(args.output_path, exist_ok=True)
    output_hdf5_path = os.path.join(args.output_path, f"{args.datatype.lower()}_instance.hdf5")
    
    process_data(nd2_paths, output_hdf5_path, args.classes, 
                patch_size=args.patch_size, overlap=args.overlap)