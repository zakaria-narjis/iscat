import os
import h5py
import numpy as np
import pandas as pd
from nd2 import ND2File
import cv2
from pathlib import Path

def get_nd2_and_csv_paths(base_path, option):
    """
    Collect paths to .nd2 files and corresponding CSV files.
    
    Args:
        base_path (str): Base directory to search
        option (str): 'Brightfield' or 'Laser'
    
    Returns:
        list: List of tuples (nd2_path, [csv_paths])
    """
    if option not in {'Brightfield', 'Laser'}:
        raise ValueError("Option must be 'Brightfield' or 'Laser'")
    
    file_pairs = []
    mask_names = ['Captured Cy5.csv', 'Captured FITC.csv', 'Captured TRITC.csv']
    
    for root, _, files in os.walk(base_path):
        if 'Metasurface' in Path(root).parts:
            target_folder = os.path.join(root, option)
            if os.path.isdir(target_folder):
                for file in os.listdir(target_folder):
                    if file.endswith('.nd2'):
                        nd2_path = os.path.join(target_folder, file)
                        csv_paths = []
                        for mask_name in mask_names:
                            csv_path = os.path.join(target_folder, mask_name)
                            if os.path.exists(csv_path):
                                csv_paths.append((csv_path, len(csv_paths)))  # Include class index
                        if csv_paths:
                            file_pairs.append((nd2_path, csv_paths))
    
    return file_pairs

def extract_particle_region(image, bbox):
    """Extract region from image using bbox coordinates."""
    xmin, xmax, ymin, ymax = bbox
    return image[:, ymin:ymax, xmin:xmax]

def average_region(region, axis='x', target_size=16):
    """
    Average the region along specified axis and handle symmetric edge padding/resizing.
    
    Args:
        region (np.ndarray): Input region of shape (Z, H, W)
        axis (str): Averaging axis ('x' or 'y')
        target_size (int): Target size for the first dimension
    
    Returns:
        np.ndarray: Processed region of shape (target_size, Z)
    """
    if axis == 'x':
        averaged = np.mean(region, axis=2)  # Average along x-axis
    else:
        averaged = np.mean(region, axis=1)  # Average along y-axis
    
    # Transpose to get (H/W, Z) shape
    averaged = averaged.T
    
    current_size = averaged.shape[0]
    if current_size == target_size:
        return averaged
    elif current_size > target_size:
        # Use interpolation to reduce size
        return cv2.resize(averaged, (averaged.shape[1], target_size))
    else:
        # Calculate padding sizes for both sides
        total_pad = target_size - current_size
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        
        # Create padded array using edge values
        padded = np.zeros((target_size, averaged.shape[1]))
        padded[pad_top:pad_top+current_size] = averaged
        
        # Fill top padding with first row
        padded[:pad_top] = averaged[0]
        # Fill bottom padding with last row
        padded[pad_top+current_size:] = averaged[-1]
        
        return padded

def radial_average(image, center, axes, target_size=None):
    """
    Perform radial averaging over an elliptical region with optional padding.
    
    Args:
        image (np.ndarray): Input image of shape (Z, H, W)
        center (tuple): (x, y) center coordinates
        axes (tuple): (x_axis, y_axis) lengths
        target_size (int): Optional target size for output profile
    
    Returns:
        np.ndarray: Radially averaged values
    """
    z, h, w = image.shape
    y, x = np.ogrid[:h, :w]
    
    # Create elliptical mask
    ellipse_mask = ((x - center[0])**2 / axes[0]**2 + 
                    (y - center[1])**2 / axes[1]**2) <= 1
    
    # Calculate distances from center
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_radius = int(np.sqrt(axes[0]**2 + axes[1]**2))
    
    radial_profile = np.zeros((z, max_radius))
    for i in range(z):
        for r in range(max_radius):
            mask = (distances >= r) & (distances < r + 1) & ellipse_mask
            if mask.any():
                radial_profile[i, r] = image[i][mask].mean()
    
    profile = radial_profile.T
    
    # Add padding if target_size is specified
    if target_size is not None and target_size > max_radius:
        total_pad = target_size - max_radius
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        
        padded = np.zeros((target_size, z))
        padded[pad_top:pad_top+max_radius] = profile
        
        # Fill top padding with first row
        padded[:pad_top] = profile[0]
        # Fill bottom padding with last row
        padded[pad_top+max_radius:] = profile[-1]
        
        return padded
    
    return profile

def process_dataset(file_pairs, output_path, averaging_axis='x', target_size=16, 
                   use_radial=False):
    """
    Process the entire dataset and save to HDF5.
    
    Args:
        file_pairs (list): List of (nd2_path, csv_paths) tuples
        output_path (str): Output HDF5 file path
        averaging_axis (str): 'x' or 'y'
        target_size (int): Target size for first dimension
        use_radial (bool): Whether to use radial averaging
    """
    with h5py.File(output_path, 'w') as hf:
        data_list = []
        labels_list = []
        
        for nd2_path, csv_paths in file_pairs:
            print(f"Processing {nd2_path}")
            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()  # Shape: (Z, H, W)
                
                for csv_path, class_idx in csv_paths:
                    df = pd.read_csv(csv_path)
                    
                    for _, row in df.iterrows():
                        bbox = (int(row['xMin']), int(row['xMax']), 
                               int(row['yMin']), int(row['yMax']))
                        region = extract_particle_region(image, bbox)
                        
                        if use_radial:
                            center = ((bbox[1] - bbox[0])//2, (bbox[3] - bbox[2])//2)
                            axes = ((bbox[1] - bbox[0])//2, (bbox[3] - bbox[2])//2)
                            processed = radial_average(region, center, axes, target_size)
                        else:
                            processed = average_region(region, averaging_axis, target_size)
                        
                        data_list.append(processed)
                        labels_list.append(class_idx)
        
        if data_list:
            data_array = np.stack(data_list)
            labels_array = np.array(labels_list)
            
            hf.create_dataset('data', data=data_array)
            hf.create_dataset('labels', data=labels_array)
            
            # Save metadata
            hf.attrs['averaging_method'] = 'radial' if use_radial else f'axis_{averaging_axis}'
            hf.attrs['target_size'] = target_size
            hf.attrs['class_info'] = 'Class 0: Cy5 (80nm), Class 1: FITC (300nm), Class 2: TRITC (1300nm)'

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process particle data and create HDF5 dataset")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], 
                       required=True, help="Data type")
    parser.add_argument("--output_path", type=str, default='dataset', help="Folder Path for the output HDF5 file.")
    parser.add_argument("--averaging_axis", type=str, choices=['x', 'y'], 
                       default='x', help="Axis for averaging")
    parser.add_argument("--target_size", type=int, default=16,
                       help="Target size for first dimension")
    parser.add_argument("--use_radial", action="store_true",
                       help="Use radial averaging instead of axis averaging")
    
    args = parser.parse_args()
    
    # Define data paths
    data_paths = [
        os.path.join('data', '2024_11_11', 'Metasurface', 'Chip_02'),
        os.path.join('data', '2024_11_12', 'Metasurface', 'Chip_01'),
        # os.path.join('data', '2024_11_29', 'Metasurface', 'Chip_02')
    ]
    
    # Collect all file pairs
    all_file_pairs = []
    for data_path in data_paths:
        all_file_pairs.extend(get_nd2_and_csv_paths(data_path, args.datatype))
    output_hdf5_path = os.path.join(args.output_path, f"{args.datatype.lower()}_particles.hdf5")
    # Process dataset
    process_dataset(
        all_file_pairs,
        output_hdf5_path,
        averaging_axis=args.averaging_axis,
        target_size=args.target_size,
        use_radial=args.use_radial
    )