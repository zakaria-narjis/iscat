import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing.utils import Utils
import os 
import argparse

def get_fluo_paths(root_path:str, mode:str="Brightfield"):
    """
    Extract paths to .nd2 files and corresponding TIFF files from the specified mode folder.

    Args:
        root_path (str): The root directory to search.
        mode (str): The folder name to focus on (default is 'Brightfield').

    Returns:
        tuple: Two lists - list of .nd2 file paths and list of tuples with corresponding TIFF file paths.
    """

    target_files = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == mode:

            # Generate TIFF file paths dynamically based on the prefix
            cy5_path = os.path.join(dirpath, f'Captured Cy5.tif')
            fitc_path = os.path.join(dirpath, f'Captured FITC.tif')
            tritc_path = os.path.join(dirpath, f'Captured TRITC.tif')
            target_files.append((cy5_path, fitc_path, tritc_path))
            
            # Ensure all three TIFF files exist
            assert all(os.path.exists(path) for path in [cy5_path, fitc_path, tritc_path])
        
            
    return target_files

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files for image and mask patches.")
    parser.add_argument("--base_path", type=str, required=True, help="The base directory to search.")
    parser.add_argument("--seg_method", type=str, choices=["comdet", "kmeans"], default="comdet", help="The segmentation method to use.")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], default="Brightfield", help="The folder to consider.")

    args = parser.parse_args()
    tif_tuples = get_fluo_paths(args.base_path, args.datatype)
    Utils.generate_np_masks(tif_tuples,seg_args=None,seg_method=args.seg_method)