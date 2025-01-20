"""
This script is used to generate the hdf5 file for the dataset. The dataset is structured as follows:
- Each HDF5 file contains two datasets: image_patches and mask_patches.
- The image_patches dataset contains image patches extracted from ND2 files.
- The mask_patches dataset contains mask patches extracted from corresponding mask files.
- The image_patches dataset has shape (N, Z, H, W), where N is the number of patches, Z is the number of slices in the ND2 file, and H, W are the patch dimensions.
- The mask_patches dataset has shape (N, C, H, W), where N is the number of patches, C is the number of masks, and H, W are the patch dimensions.
- Mask classes are consistently ordered:
  - Class 0: Cy5 (80nm)
  - Class 1: FITC (300nm)
  - Class 2: TRITC (1300nm)
"""
import os
import h5py
import numpy as np
from nd2 import ND2File
import argparse

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

def nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=(256, 256), overlap=0):
    """
    Load ND2 files, extract image and mask patches, and save them into an HDF5 file with metadata.

    Args:
        nd2_paths (list of str): Paths to ND2 files.
        output_hdf5_path (str): Path to the output HDF5 file.
        patch_size (tuple): Size of the patches (height, width).
        overlap (int): Overlap between patches in pixels.

    Returns:
        None
    """
    patch_height, patch_width = patch_size

    # Define the consistent order of masks and their metadata
    mask_order = [
        "Captured Cy5_mask.npy",    # Class 0
        "Captured FITC_mask.npy",   # Class 1
        "Captured TRITC_mask.npy"   # Class 2
    ]
    
    mask_metadata = {
        "Captured Cy5_mask.npy": "Cy5: 80nm",
        "Captured FITC_mask.npy": "FITC: 300nm",
        "Captured TRITC_mask.npy": "TRITC: 1300nm",
    }

    # Mapping for the 2024_11_29 case
    rename_mapping = {
        "Captured Cy5_mask.npy": "Captured FITC_mask.npy",
        "Captured FITC_mask.npy": "Captured RITC_mask.npy",
        "Captured TRITC_mask.npy": "Captured Cy5_mask.npy",
    }

    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        image_dataset = None
        mask_dataset = None

        hdf5_file.attrs["description"] = "Image and mask patches with nanometer scale metadata."
        hdf5_file.attrs["mask_info"] = ", ".join([f"class:{idx} ({mask_metadata[mask]})" for idx, mask in enumerate(mask_order)])

        for nd2_path in nd2_paths:
            print(f"Processing {nd2_path}...")

            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()
                if image.ndim != 3:
                    raise ValueError(f"Expected 3D data (Z, H, W), got shape {image.shape}")
                num_slices, height, width = image.shape

            nd2_dir = os.path.dirname(nd2_path)
            
            # Handle the special case for 2024_11_29
            is_special_case = "2024_11_29" in nd2_path
            if is_special_case:
                print('Found 2024_11_29 in the path. Using remapped mask names...')
                mask_paths = {original: os.path.join(nd2_dir, rename_mapping[original]) for original in mask_order}
            else:
                mask_paths = {mask: os.path.join(nd2_dir, mask) for mask in mask_order}

            # Validate mask files
            for mask_name, mask_path in mask_paths.items():
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file {mask_path} not found.")

            # Load masks in the correct order
            masks = {mask_name: np.load(mask_paths[mask_name]) for mask_name in mask_order}

            # Validate mask dimensions
            for mask_name, mask_array in masks.items():
                if mask_array.shape != (height, width):
                    raise ValueError(f"Mask {mask_name} shape {mask_array.shape} does not match image shape {(height, width)}")

            # Extract patches
            for y in range(0, height - patch_height + 1, patch_height - overlap):
                for x in range(0, width - patch_width + 1, patch_width - overlap):
                    image_patch = image[:, y:y + patch_height, x:x + patch_width]
                    
                    # Extract mask patches in the defined order
                    mask_patches = [masks[mask_name][y:y + patch_height, x:x + patch_width] 
                                  for mask_name in mask_order]

                    # Initialize or resize datasets
                    if image_dataset is None:
                        image_dataset = hdf5_file.create_dataset(
                            "image_patches",
                            shape=(0, num_slices, patch_height, patch_width),
                            maxshape=(None, num_slices, patch_height, patch_width),
                            chunks=(1, num_slices, patch_height, patch_width),
                            dtype=image_patch.dtype
                        )

                    if mask_dataset is None:
                        mask_dataset = hdf5_file.create_dataset(
                            "mask_patches",
                            shape=(0, len(mask_order), patch_height, patch_width),
                            maxshape=(None, len(mask_order), patch_height, patch_width),
                            chunks=(1, len(mask_order), patch_height, patch_width),
                            dtype=np.uint8
                        )
                        # Add metadata with consistent class ordering
                        for idx, mask_name in enumerate(mask_order):
                            mask_dataset.attrs[f"mask_{idx}"] = f"Class {idx}: {mask_metadata[mask_name]}"

                    # Save patches
                    image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                    image_dataset[-1] = image_patch

                    mask_dataset.resize(mask_dataset.shape[0] + 1, axis=0)
                    mask_dataset[-1] = np.stack(mask_patches)

    print(f"Image and mask patches saved to {output_hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files for image and mask patches.")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], required=True, help="Specify the data type (Brightfield or Laser).")
    parser.add_argument("--output_path", type=str, default='dataset', help="Folder Path for the output HDF5 file.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256), help="Patch size (height, width). Default is 256x256.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between patches in pixels. Default is 0.")
    args = parser.parse_args()

    data_path_1 = os.path.join('data', '2024_11_11', 'Metasurface', 'Chip_02')
    data_path_2 = os.path.join('data', '2024_11_12', 'Metasurface', 'Chip_01')
    data_path_3 = os.path.join('data', '2024_11_29', 'Metasurface', 'Chip_02')
    
    nd2_paths = []
    for data_path in [data_path_1, data_path_2, data_path_3]:
        nd2_paths.extend(get_nd2_paths(data_path, args.datatype))
        
    output_hdf5_path = os.path.join(args.output_path, f"{args.datatype.lower()}.hdf5")
    nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=args.patch_size, overlap=args.overlap)