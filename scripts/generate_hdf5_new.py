import os
import h5py
import numpy as np
import pandas as pd
from nd2 import ND2File
import argparse
from scipy.ndimage import binary_fill_holes

def get_nd2_paths(base_path, option):
    """
    Recursively collects paths to .nd2 files inside specified subfolders of Metasurface directories.
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

def create_instance_mask(bbox_data, shape, class_id):
    """
    Create instance masks using ellipses for each bounding box.
    
    Args:
        bbox_data (pd.DataFrame): DataFrame containing bounding box information
        shape (tuple): Shape of the output mask (height, width)
        class_id (int): Class ID for the particles
        
    Returns:
        tuple: (instance_masks, bboxes, class_labels)
    """
    instance_masks = []
    bboxes = []
    class_labels = []
    
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    for _, row in bbox_data.iterrows():
        center_x = int((row['xMin'] + row['xMax']) / 2)
        center_y = int((row['yMin'] + row['yMax']) / 2)
        axes_x = max(int((row['xMax'] - row['xMin']) / 2), 1)
        axes_y = max(int((row['yMax'] - row['yMin']) / 2), 1)
        
        # Create ellipse mask
        ellipse_mask = ((x - center_x) / axes_x)**2 + ((y - center_y) / axes_y)**2 <= 1
        ellipse_mask = binary_fill_holes(ellipse_mask).astype(np.uint8)
        
        instance_masks.append(ellipse_mask)
        bboxes.append([row['xMin'], row['yMin'], row['xMax'], row['yMax']])
        class_labels.append(class_id)
    
    return instance_masks, bboxes, class_labels

def load_bbox_data(nd2_dir, mask_name):
    """
    Load bounding box data from CSV files.
    """
    csv_name = mask_name.replace('_mask.npy', '.csv')
    csv_path = os.path.join(nd2_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found.")
    return pd.read_csv(csv_path)

def extract_patch_instances(bbox_data, patch_coords, class_id):
    """
    Extract bounding boxes that fall within the patch coordinates.
    """
    x_min, y_min, x_max, y_max = patch_coords
    
    # Filter bounding boxes that overlap with the patch
    patch_bboxes = bbox_data[
        (bbox_data['xMin'] < x_max) & 
        (bbox_data['xMax'] > x_min) & 
        (bbox_data['yMin'] < y_max) & 
        (bbox_data['yMax'] > y_min)
    ].copy()
    
    # Adjust coordinates relative to patch
    patch_bboxes['xMin'] = (patch_bboxes['xMin'] - x_min).clip(lower=0)
    patch_bboxes['yMin'] = (patch_bboxes['yMin'] - y_min).clip(lower=0)
    patch_bboxes['xMax'] = (patch_bboxes['xMax'] - x_min).clip(upper=x_max-x_min)
    patch_bboxes['yMax'] = (patch_bboxes['yMax'] - y_min).clip(upper=y_max-y_min)
    
    return patch_bboxes

def nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=(256, 256), overlap=0):
    """
    Load ND2 files, extract image and mask patches, and save them into an HDF5 file with metadata.
    Now includes instance segmentation information.
    """
    patch_height, patch_width = patch_size

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

    rename_mapping = {
        "Captured Cy5_mask.npy": "Captured FITC_mask.npy",
        "Captured FITC_mask.npy": "Captured TRITC_mask.npy",
        "Captured TRITC_mask.npy": "Captured Cy5_mask.npy",
    }

    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        # Initialize datasets
        image_dataset = None
        mask_dataset = None  # For semantic segmentation
        instance_dataset = None  # For instance segmentation
        bbox_dataset = None
        label_dataset = None

        hdf5_file.attrs["description"] = "Image and mask patches with instance segmentation data."
        hdf5_file.attrs["mask_info"] = ", ".join([f"class:{idx} ({mask_metadata[mask]})" for idx, mask in enumerate(mask_order)])

        for nd2_path in nd2_paths:
            print(f"Processing {nd2_path}...")
            
            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()
                if image.ndim != 3:
                    raise ValueError(f"Expected 3D data (Z, H, W), got shape {image.shape}")
                num_slices, height, width = image.shape

            nd2_dir = os.path.dirname(nd2_path)
            is_special_case = "2024_11_29" in nd2_path
            
            if is_special_case:
                mask_paths = {original: os.path.join(nd2_dir, rename_mapping[original]) for original in mask_order}
                bbox_paths = {original: os.path.join(nd2_dir, rename_mapping[original].replace('_mask.npy', '.csv')) 
                            for original in mask_order}
            else:
                mask_paths = {mask: os.path.join(nd2_dir, mask) for mask in mask_order}
                bbox_paths = {mask: os.path.join(nd2_dir, mask.replace('_mask.npy', '.csv')) 
                            for mask in mask_order}

            # Load masks and bbox data
            masks = {mask_name: np.load(mask_paths[mask_name]) for mask_name in mask_order}
            bbox_data = {mask_name: load_bbox_data(nd2_dir, mask_name) for mask_name in mask_order}

            # Extract patches
            for y in range(0, height - patch_height + 1, patch_height - overlap):
                for x in range(0, width - patch_width + 1, patch_width - overlap):
                    patch_coords = (x, y, x + patch_width, y + patch_height)
                    image_patch = image[:, y:y + patch_height, x:x + patch_width]
                    
                    # Extract semantic segmentation masks
                    mask_patches = [masks[mask_name][y:y + patch_height, x:x + patch_width] 
                                  for mask_name in mask_order]
                    
                    # Process instance segmentation data
                    all_instance_masks = []
                    all_bboxes = []
                    all_labels = []
                    
                    for class_id, mask_name in enumerate(mask_order):
                        # Filter bounding boxes for this patch
                        patch_bbox_data = extract_patch_instances(bbox_data[mask_name], patch_coords, class_id)
                        
                        if not patch_bbox_data.empty:
                            instance_masks, bboxes, labels = create_instance_mask(
                                patch_bbox_data, 
                                (patch_height, patch_width),
                                class_id
                            )
                            all_instance_masks.extend(instance_masks)
                            all_bboxes.extend(bboxes)
                            all_labels.extend(labels)

                    # Initialize or resize datasets
                    if image_dataset is None:
                        image_dataset = hdf5_file.create_dataset(
                            "image_patches",
                            shape=(0, num_slices, patch_height, patch_width),
                            maxshape=(None, num_slices, patch_height, patch_width),
                            chunks=(1, num_slices, patch_height, patch_width),
                            dtype=image_patch.dtype
                        )

                        mask_dataset = hdf5_file.create_dataset(
                            "mask_patches",
                            shape=(0, len(mask_order), patch_height, patch_width),
                            maxshape=(None, len(mask_order), patch_height, patch_width),
                            chunks=(1, len(mask_order), patch_height, patch_width),
                            dtype=np.uint8
                        )

                        # Add instance segmentation datasets
                        instance_dataset = hdf5_file.create_dataset(
                            "instance_masks",
                            shape=(0, patch_height, patch_width),
                            maxshape=(None, patch_height, patch_width),
                            chunks=(1, patch_height, patch_width),
                            dtype=np.uint8
                        )

                        bbox_dataset = hdf5_file.create_dataset(
                            "bboxes",
                            shape=(0, 4),  # [xmin, ymin, xmax, ymax]
                            maxshape=(None, 4),
                            chunks=(1, 4),
                            dtype=np.float32
                        )

                        label_dataset = hdf5_file.create_dataset(
                            "class_labels",
                            shape=(0,),
                            maxshape=(None,),
                            chunks=(1,),
                            dtype=np.int32
                        )

                    # Save patches and instance data
                    image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                    image_dataset[-1] = image_patch

                    mask_dataset.resize(mask_dataset.shape[0] + 1, axis=0)
                    mask_dataset[-1] = np.stack(mask_patches)

                    # Save instance segmentation data
                    if all_instance_masks:
                        current_size = instance_dataset.shape[0]
                        new_size = current_size + len(all_instance_masks)
                        
                        instance_dataset.resize(new_size, axis=0)
                        instance_dataset[current_size:] = np.stack(all_instance_masks)
                        
                        bbox_dataset.resize(new_size, axis=0)
                        bbox_dataset[current_size:] = np.array(all_bboxes)
                        
                        label_dataset.resize(new_size, axis=0)
                        label_dataset[current_size:] = np.array(all_labels)

    print(f"Image, mask patches, and instance segmentation data saved to {output_hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files for image and mask patches with instance segmentation.")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], required=True,
                      help="Specify the data type (Brightfield or Laser).")
    parser.add_argument("--output_path", type=str, default='dataset',
                      help="Folder Path for the output HDF5 file.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(256, 256),
                      help="Patch size (height, width). Default is 256x256.")
    parser.add_argument("--overlap", type=int, default=0,
                      help="Overlap between patches in pixels. Default is 0.")
    args = parser.parse_args()

    data_path_1 = os.path.join('data', '2024_11_11', 'Metasurface', 'Chip_02')
    data_path_2 = os.path.join('data', '2024_11_12', 'Metasurface', 'Chip_01')
    # data_path_3 = os.path.join('data', '2024_11_29', 'Metasurface', 'Chip_02')
    nd2_paths = []
    for data_path in [data_path_1, data_path_2]:
        nd2_paths.extend(get_nd2_paths(data_path, args.datatype))
        
    output_hdf5_path = os.path.join(args.output_path, f"{args.datatype.lower()}_instance.hdf5")
    nd2_to_hdf5(nd2_paths, output_hdf5_path, patch_size=args.patch_size, overlap=args.overlap)