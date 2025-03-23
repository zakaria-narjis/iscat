import os
import h5py
import numpy as np
import pandas as pd
import tifffile
import argparse
import glob


def create_mask_from_csv(csv_path, image_shape):
    """
    Create a binary mask from bounding box coordinates in a CSV file,
    using ellipses inside the bounding boxes.
    
    Args:
        csv_path (str): Path to the CSV file containing bounding box coordinates.
        image_shape (tuple): Shape of the image (height, width).
        
    Returns:
        np.ndarray: Binary mask with the same shape as the image.
    """
    # Initialize an empty mask
    label_canvas = np.zeros(image_shape, dtype=np.uint8)
    
    # Precompute y, x coordinates
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]
    
    # Read CSV file containing bounding box coordinates
    if os.path.exists(csv_path):
        try:
            # Load the CSV data
            particles_positions_df = pd.read_csv(csv_path)
            
            # Process each bounding box and add ellipse to the mask
            for _, row in particles_positions_df.iterrows():
                # Compute ellipse parameters
                # Adapt column names based on your CSV format
                try:
                    # First try with xMin, xMax format
                    center_x = int((row['xMin'] + row['xMax']) / 2)
                    center_y = int((row['yMin'] + row['yMax']) / 2)
                    axes_x = max(1, int((row['xMax'] - row['xMin']) / 2))  # Ensure non-zero
                    axes_y = max(1, int((row['yMax'] - row['yMin']) / 2))  # Ensure non-zero
                except KeyError:
                    # Try with x, y, width, height format
                    try:
                        center_x = int(row['x'] + row['width'] / 2)
                        center_y = int(row['y'] + row['height'] / 2)
                        axes_x = max(1, int(row['width'] / 2))  # Ensure non-zero
                        axes_y = max(1, int(row['height'] / 2))  # Ensure non-zero
                    except KeyError:
                        print(f"Warning: CSV format not recognized. Columns available: {particles_positions_df.columns.tolist()}")
                        continue
                
                # Vectorized ellipse mask creation
                ellipse_mask = ((x - center_x) / axes_x)**2 + ((y - center_y) / axes_y)**2 <= 1
                
                # Efficient overlap and labeling
                label_canvas[ellipse_mask] = 1
                
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    else:
        print(f"Warning: CSV file {csv_path} not found.")
    
    return label_canvas


def tif_to_hdf5(data_path, output_hdf5_path, patch_size=(256, 256), overlap=0):
    """
    Load TIF files, extract image patches, create elliptical masks from CSV files, 
    and save them into an HDF5 file.

    Args:
        data_path (str): Path to the directory containing TIF and CSV files.
        output_hdf5_path (str): Path to the output HDF5 file.
        patch_size (tuple): Size of the patches (height, width).
        overlap (int): Overlap between patches in pixels.

    Returns:
        None
    """
    patch_height, patch_width = patch_size
    
    # Find all TIF files in the data path
    all_tif_files = sorted(glob.glob(os.path.join(data_path, "*.tif")))
    
    # Filter to only include files matching the pattern (##.tif)
    tif_files = []
    for file_path in all_tif_files:
        file_name = os.path.basename(file_path)
        # Check if the filename matches the pattern (digits only before .tif)
        if file_name.split('.')[0].isdigit():
            tif_files.append(file_path)
    
    if not tif_files:
        raise FileNotFoundError(f"No numbered TIF files (e.g., 01.tif) found in {data_path}")
    
    print(f"Found {len(tif_files)} numbered TIF files.")
    
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        image_dataset = None
        mask_dataset = None
        
        # Set metadata
        hdf5_file.attrs["description"] = "Image and mask patches with nanometer scale metadata."
        hdf5_file.attrs["mask_info"] = "class:0 (FITC: 80nm)"
        
        for tif_file in tif_files:
            file_basename = os.path.basename(tif_file).split('.')[0]  # e.g., "01" from "01.tif"
            print(f"Processing {tif_file}...")
            
            # Corresponding CSV file path
            csv_file = os.path.join(data_path, f"{file_basename}_FITC.csv")
            
            # Load TIF image
            try:
                # First try with tifffile which handles multi-page TIFs better
                image = tifffile.imread(tif_file)
                
                # Check if the image is multi-slice (3D) or single-slice (2D)
                if image.ndim == 2:  # Convert 2D image to 3D with single slice
                    image = np.expand_dims(image, axis=0)
                elif image.ndim != 3:
                    raise ValueError(f"Unexpected image dimensions: {image.shape}")
                
                num_slices, height, width = image.shape
                
            except Exception as e:
                print(f"Error loading image {tif_file}: {e}")
                continue
            
            # Create elliptical mask from CSV
            mask = create_mask_from_csv(csv_file, (height, width))
            
            # Extract patches
            for y in range(0, height - patch_height + 1, patch_height - overlap):
                for x in range(0, width - patch_width + 1, patch_width - overlap):
                    # Extract image patch
                    image_patch = image[:, y:y + patch_height, x:x + patch_width]
                    
                    # Extract mask patch
                    mask_patch = mask[y:y + patch_height, x:x + patch_width]
                    
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
                        # We now have a single class
                        mask_dataset = hdf5_file.create_dataset(
                            "mask_patches",
                            shape=(0, 1, patch_height, patch_width),
                            maxshape=(None, 1, patch_height, patch_width),
                            chunks=(1, 1, patch_height, patch_width),
                            dtype=np.uint8
                        )
                        # Add metadata for the single mask class
                        mask_dataset.attrs["mask_0"] = "Class 0: FITC (80nm)"
                    
                    # Save patches
                    image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                    image_dataset[-1] = image_patch
                    
                    mask_dataset.resize(mask_dataset.shape[0] + 1, axis=0)
                    # Need to add an extra dimension for the single class
                    mask_dataset[-1] = np.expand_dims(mask_patch, axis=0)
    
    print(f"Image and mask patches saved to {output_hdf5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files for TIF image and CSV mask patches.")
    parser.add_argument("--datatype", type=str, default="Brightfield",choices=["Brightfield", "Laser"], help="Specify the data type (Brightfield or Laser).")
    parser.add_argument("--output_path", type=str, default='dataset', help="Folder Path for the output HDF5 file.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256), help="Patch size (height, width). Default is 256x256.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between patches in pixels. Default is 0.")
    args = parser.parse_args()

    # Path to the data folder
    data_path = os.path.join('dataset_', '2025_03_04')
    
    # Output HDF5 path
    output_hdf5_path = os.path.join(args.output_path, f"EV_{args.datatype.lower()}.hdf5")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    tif_to_hdf5(data_path, output_hdf5_path, patch_size=args.patch_size, overlap=args.overlap)