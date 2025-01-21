import os
import h5py
import numpy as np
import pandas as pd
from nd2 import ND2File
import argparse

def get_nd2_paths(base_path, option):
    """Same as before"""
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

def create_instance_mask(canvas_shape, bbox_data, class_id):
    """
    Create instance segmentation mask for particles within a patch.
    
    Args:
        canvas_shape (tuple): Shape of the canvas (height, width)
        bbox_data (DataFrame): DataFrame containing bounding box information
        class_id (int): Class ID for the current particle type
        
    Returns:
        numpy.ndarray: Instance segmentation mask where each instance has a unique ID
    """
    instance_canvas = np.zeros(canvas_shape, dtype=np.uint8)
    y, x = np.ogrid[:canvas_shape[0], :canvas_shape[1]]
    
    for instance_id, row in enumerate(bbox_data.itertuples(), start=1):
        center_x = int((row.xMin + row.xMax) / 2)
        center_y = int((row.yMin + row.yMax) / 2)
        axes_x = max(int((row.xMax - row.xMin) / 2), 1)
        axes_y = max(int((row.yMax - row.yMin) / 2), 1)
        
        # Create ellipse mask
        ellipse_mask = ((x - center_x) / axes_x)**2 + ((y - center_y) / axes_y)**2 <= 1
        instance_canvas[ellipse_mask] = instance_id
    
    return instance_canvas

def get_instances_in_patch(bbox_df, patch_coords, patch_size):
    """
    Filter instances that fall within the current patch.
    
    Args:
        bbox_df (DataFrame): DataFrame containing bounding box information
        patch_coords (tuple): Top-left coordinates of the patch (x, y)
        patch_size (tuple): Size of the patch (height, width)
        
    Returns:
        DataFrame: Filtered DataFrame containing only instances within the patch
    """
    x, y = patch_coords
    h, w = patch_size
    
    # Adjust coordinates to be relative to the patch
    patch_instances = bbox_df[
        (bbox_df['xMin'] >= x) & 
        (bbox_df['xMax'] <= x + w) &
        (bbox_df['yMin'] >= y) & 
        (bbox_df['yMax'] <= y + h)
    ].copy()
    
    if not patch_instances.empty:
        patch_instances['xMin'] = patch_instances['xMin'] - x
        patch_instances['xMax'] = patch_instances['xMax'] - x
        patch_instances['yMin'] = patch_instances['yMin'] - y
        patch_instances['yMax'] = patch_instances['yMax'] - y
    
    return patch_instances

def nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=(256, 256), overlap=0):
    """
    Load ND2 files, extract image and instance masks, and save them into an HDF5 file.
    
    Args:
        nd2_paths (list): Paths to ND2 files
        output_hdf5_path (str): Output HDF5 file path
        patch_size (tuple): Size of patches (height, width)
        overlap (int): Overlap between patches
    """
    patch_height, patch_width = patch_size
    
    # Define particle classes and their metadata
    particle_classes = [
        ("Captured Cy5", 0, "80nm"),
        ("Captured FITC", 1, "300nm"),
        ("Captured TRITC", 2, "1300nm")
    ]
    
    # Special case handling for 2024_11_29
    rename_mapping = {
        "Captured Cy5": "Captured FITC",
        "Captured FITC": "Captured TRITC",
        "Captured TRITC": "Captured Cy5"
    }
    
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        # Initialize datasets
        image_dataset = None
        instance_dataset = None  # Will store instance masks
        class_dataset = None     # Will store class labels for each instance
        
        # Add metadata
        hdf5_file.attrs["description"] = "Image patches with instance segmentation masks"
        hdf5_file.attrs["class_info"] = ", ".join([f"class:{idx} ({name}: {size})" for name, idx, size in particle_classes])
        
        for nd2_path in nd2_paths:
            print(f"Processing {nd2_path}...")
            
            # Load image data
            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()
                if image.ndim != 3:
                    raise ValueError(f"Expected 3D data (Z, H, W), got shape {image.shape}")
                num_slices, height, width = image.shape
            
            nd2_dir = os.path.dirname(nd2_path)
            is_special_case = "2024_11_29" in nd2_path
            
            # Load bbox data for all classes
            bbox_data = {}
            for particle_name, class_id, _ in particle_classes:
                actual_name = rename_mapping[particle_name] if is_special_case else particle_name
                csv_path = os.path.join(nd2_dir, f"{actual_name}.csv")
                if os.path.exists(csv_path):
                    bbox_data[class_id] = pd.read_csv(csv_path)
                else:
                    print(f"Warning: {csv_path} not found")
                    bbox_data[class_id] = pd.DataFrame()
            
            # Extract patches
            for y in range(0, height - patch_height + 1, patch_height - overlap):
                for x in range(0, width - patch_width + 1, patch_width - overlap):
                    # Extract image patch
                    image_patch = image[:, y:y + patch_height, x:x + patch_width]
                    
                    # Create instance masks for each class
                    instance_masks = []
                    for class_id in range(len(particle_classes)):
                        if class_id in bbox_data:
                            patch_instances = get_instances_in_patch(
                                bbox_data[class_id], 
                                (x, y), 
                                (patch_height, patch_width)
                            )
                            instance_mask = create_instance_mask(
                                (patch_height, patch_width), 
                                patch_instances, 
                                class_id
                            )
                            instance_masks.append(instance_mask)
                        else:
                            instance_masks.append(np.zeros((patch_height, patch_width), dtype=np.uint8))
                    
                    # Stack instance masks
                    stacked_instances = np.stack(instance_masks)
                    
                    # Initialize or resize datasets
                    if image_dataset is None:
                        image_dataset = hdf5_file.create_dataset(
                            "image_patches",
                            shape=(0, num_slices, patch_height, patch_width),
                            maxshape=(None, num_slices, patch_height, patch_width),
                            chunks=(1, num_slices, patch_height, patch_width),
                            dtype=image_patch.dtype
                        )
                        
                        instance_dataset = hdf5_file.create_dataset(
                            "instance_masks",
                            shape=(0, len(particle_classes), patch_height, patch_width),
                            maxshape=(None, len(particle_classes), patch_height, patch_width),
                            chunks=(1, len(particle_classes), patch_height, patch_width),
                            dtype=np.uint8
                        )
                    
                    # Save patches and masks
                    image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                    image_dataset[-1] = image_patch
                    
                    instance_dataset.resize(instance_dataset.shape[0] + 1, axis=0)
                    instance_dataset[-1] = stacked_instances

    print(f"Dataset saved to {output_hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files with instance segmentation masks.")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], required=True)
    parser.add_argument("--output_path", type=str, default='dataset')
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()

    data_paths = [
        os.path.join('data', '2024_11_11', 'Metasurface', 'Chip_02'),
        os.path.join('data', '2024_11_12', 'Metasurface', 'Chip_01'),
        # os.path.join('data', '2024_11_29', 'Metasurface', 'Chip_02')
    ]
    
    nd2_paths = []
    for data_path in data_paths:
        nd2_paths.extend(get_nd2_paths(data_path, args.datatype))
        
    output_hdf5_path = os.path.join(args.output_path, f"{args.datatype.lower()}_instance.hdf5")
    nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=args.patch_size, overlap=args.overlap)