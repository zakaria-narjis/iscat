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

def load_bbox_data(csv_path: str, class_id: int) -> pd.DataFrame:
    """
    Load and process bounding box data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing bounding box information
        class_id (int): Class ID for this particle type
        
    Returns:
        DataFrame: Processed bbox data with columns [xmin, ymin, xmax, ymax, class_id]
    """
    df = pd.read_csv(csv_path)
    bbox_data = df[['xMin', 'yMin', 'xMax', 'yMax']].copy()
    bbox_data['class_id'] = class_id  # Add class ID for multi-class detection
    return bbox_data

def get_patch_boxes(
    bbox_data_list: List[pd.DataFrame],
    patch_coords: Tuple[int, int, int, int],
    patch_size: Tuple[int, int]
) -> np.ndarray:
    """
    Get boxes that fall within the patch coordinates for all classes.
    
    Args:
        bbox_data_list: List of DataFrames with bounding box information for each class
        patch_coords: (y_start, y_end, x_start, x_end) coordinates of the patch
        patch_size: (height, width) of the patch
    
    Returns:
        Array of [num_instances, 5] in format [x1, y1, x2, y2, class_id]
    """
    y_start, y_end, x_start, x_end = patch_coords
    patch_height, patch_width = patch_size
    
    all_boxes = []
    
    for bbox_data in bbox_data_list:
        # Filter boxes that intersect with the current patch
        relevant_boxes = bbox_data[
            (bbox_data['xMax'] >= x_start) & 
            (bbox_data['xMin'] < x_end) & 
            (bbox_data['yMax'] >= y_start) & 
            (bbox_data['yMin'] < y_end)
        ]
        
        for _, box in relevant_boxes.iterrows():
            # Convert global coordinates to patch coordinates and normalize
            x1 = max(0, box['xMin'] - x_start) / patch_width
            y1 = max(0, box['yMin'] - y_start) / patch_height
            x2 = min(patch_width, box['xMax'] - x_start) / patch_width
            y2 = min(patch_height, box['yMax'] - y_start) / patch_height
            class_id = box['class_id']
            
            all_boxes.append([x1, y1, x2, y2, class_id])
    
    return np.array(all_boxes, dtype=np.float32) if all_boxes else np.zeros((0, 5), dtype=np.float32)

def create_instance_masks(
    semantic_masks: List[np.ndarray],
    bbox_data_list: List[pd.DataFrame],
    patch_coords: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Create instance mask for a given patch using bounding box data for all classes.
    
    Args:
        semantic_masks: List of binary semantic masks for each class
        bbox_data_list: List of bounding box data for each class
        patch_coords: (y_start, y_end, x_start, x_end) coordinates of the patch
        
    Returns:
        Instance mask where each unique value represents a different instance
    """
    y_start, y_end, x_start, x_end = patch_coords
    H, W = semantic_masks[0].shape
    instance_mask = np.zeros((H, W), dtype=np.int32)
    
    instance_id = 1  # Start from 1, 0 is background
    
    for class_idx, (semantic_mask, bbox_data) in enumerate(zip(semantic_masks, bbox_data_list)):
        # Filter boxes that intersect with the current patch
        relevant_boxes = bbox_data[
            (bbox_data['xMax'] >= x_start) & 
            (bbox_data['xMin'] < x_end) & 
            (bbox_data['yMax'] >= y_start) & 
            (bbox_data['yMin'] < y_end)
        ]
        
        for _, box in relevant_boxes.iterrows():
            # Convert global coordinates to patch coordinates
            x1 = max(0, int(box['xMin'] - x_start))
            y1 = max(0, int(box['yMin'] - y_start))
            x2 = min(W, int(box['xMax'] - x_start))
            y2 = min(H, int(box['yMax'] - y_start))
            
            # Create instance mask only where semantic mask is 1
            box_region = semantic_mask[y1:y2, x1:x2]
            instance_mask[y1:y2, x1:x2][box_region == 1] = instance_id
            instance_id += 1
    
    return instance_mask

def nd2_to_hdf5(nd2_paths: List[str], output_hdf5_path: str, patch_size=(256, 256), overlap=0):
    """
    Create dataset for multi-class instance segmentation.
    """
    patch_height, patch_width = patch_size
    
    mask_order = [
        "Captured Cy5_mask.npy",    # Class 0: Cy5 (80nm)
        "Captured FITC_mask.npy",   # Class 1: FITC (300nm)
        "Captured TRITC_mask.npy"   # Class 2: TRITC (1300nm)
    ]
    
    bbox_files = [path.replace('_mask.npy', '_mask.csv') for path in mask_order]
    
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        image_dataset = None
        semantic_dataset = None
        instance_dataset = None
        bbox_dataset = None
        
        for nd2_path in nd2_paths:
            print(f"Processing {nd2_path}...")
            
            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()
                num_slices, height, width = image.shape
            
            nd2_dir = os.path.dirname(nd2_path)
            
            # Load semantic masks and bbox data
            semantic_masks = []
            bbox_data_list = []
            
            for class_idx, (mask_name, bbox_file) in enumerate(zip(mask_order, bbox_files)):
                mask_path = os.path.join(nd2_dir, mask_name)
                bbox_path = os.path.join(nd2_dir, bbox_file)
                
                semantic_masks.append(np.load(mask_path))
                bbox_data_list.append(load_bbox_data(bbox_path, class_id=class_idx))
            
            # Extract patches
            for y in range(0, height - patch_height + 1, patch_height - overlap):
                for x in range(0, width - patch_width + 1, patch_width - overlap):
                    # Extract patches
                    image_patch = image[:, y:y + patch_height, x:x + patch_width]
                    semantic_patches = [
                        mask[y:y + patch_height, x:x + patch_width]
                        for mask in semantic_masks
                    ]
                    
                    # Get boxes and create instance masks
                    patch_coords = (y, y + patch_height, x, x + patch_width)
                    boxes = get_patch_boxes(bbox_data_list, patch_coords, patch_size)
                    instance_mask = create_instance_masks(semantic_patches, bbox_data_list, patch_coords)
                    
                    # Initialize datasets
                    if image_dataset is None:
                        image_dataset = hdf5_file.create_dataset(
                            "images",
                            shape=(0, num_slices, patch_height, patch_width),
                            maxshape=(None, num_slices, patch_height, patch_width),
                            chunks=(1, num_slices, patch_height, patch_width),
                            dtype=image_patch.dtype
                        )
                        
                        semantic_dataset = hdf5_file.create_dataset(
                            "semantic_masks",
                            shape=(0, len(mask_order), patch_height, patch_width),
                            maxshape=(None, len(mask_order), patch_height, patch_width),
                            chunks=(1, len(mask_order), patch_height, patch_width),
                            dtype=np.uint8
                        )
                        
                        instance_dataset = hdf5_file.create_dataset(
                            "instance_masks",
                            shape=(0, patch_height, patch_width),
                            maxshape=(None, patch_height, patch_width),
                            chunks=(1, patch_height, patch_width),
                            dtype=np.int32
                        )
                        
                        bbox_dataset = hdf5_file.create_dataset(
                            "boxes",
                            shape=(0, 0, 5),  # [x1, y1, x2, y2, class_id]
                            maxshape=(None, None, 5),
                            chunks=(1, 10, 5),  # Chunk size optimized for typical number of instances
                            dtype=np.float32
                        )
                    
                    # Save data
                    image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                    image_dataset[-1] = image_patch
                    
                    semantic_dataset.resize(semantic_dataset.shape[0] + 1, axis=0)
                    semantic_dataset[-1] = np.stack(semantic_patches)
                    
                    instance_dataset.resize(instance_dataset.shape[0] + 1, axis=0)
                    instance_dataset[-1] = instance_mask
                    
                    bbox_dataset.resize((bbox_dataset.shape[0] + 1, boxes.shape[0], 5))
                    bbox_dataset[-1] = boxes
        
        # Add metadata
        hdf5_file.attrs['description'] = 'Multi-class instance segmentation dataset'
        hdf5_file.attrs['num_classes'] = len(mask_order)
        hdf5_file.attrs['class_names'] = [
            'Cy5 (80nm)',
            'FITC (300nm)',
            'TRITC (1300nm)'
        ]

    print(f"Dataset saved to {output_hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files for image and mask patches.")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], required=True, help="Specify the data type (Brightfield or Laser).")
    parser.add_argument("--output_path", type=str, default='dataset', help="Folder Path for the output HDF5 file.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256), help="Patch size (height, width). Default is 256x256.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between patches in pixels. Default is 0.")
    args = parser.parse_args()

    data_path_1 = os.path.join('data', '2024_11_11', 'Metasurface', 'Chip_02')
    data_path_2 = os.path.join('data', '2024_11_12', 'Metasurface', 'Chip_01')
    # data_path_3 = os.path.join('data', '2024_11_29', 'Metasurface', 'Chip_02')
    
    nd2_paths = []
    for data_path in [data_path_1, data_path_2]:
        nd2_paths.extend(get_nd2_paths(data_path, args.datatype))
        
    output_hdf5_path = os.path.join(args.output_path, f"{args.datatype.lower()}.hdf5")
    nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=args.patch_size, overlap=args.overlap)