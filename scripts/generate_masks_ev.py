import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing.utils import Utils
import os 
import argparse
"""
Not working due to ImageJ not working as expected/ refer to notebooks/labeling_script.ipynb for generating masks
"""
def get_fluo_paths(folder_path, mode="Brightfield"):
    # out = [
    #     os.path.join(folder_path, f)
    #     for f in os.listdir(folder_path)
    #     if f.endswith('tif') and "FITC" in f
    # ]
    out = []
    for f in os.listdir(folder_path):
        if f.endswith('tif') and "FITC" in f:
            out.append((os.path.join(folder_path, f),))

    return out

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 files for image and mask patches.")
    parser.add_argument("--base_path", type=str, required=True, help="The base directory to search.")
    parser.add_argument("--seg_method", type=str, choices=["comdet", "kmeans"], default="comdet", help="The segmentation method to use.")
    parser.add_argument("--datatype", type=str, choices=["Brightfield", "Laser"], default="Brightfield", help="The folder to consider.")

    args = parser.parse_args()
    tif_tuples = get_fluo_paths(args.base_path, args.datatype)
    print(tif_tuples)
    Utils.generate_np_masks(tif_tuples,seg_args=None,seg_method=args.seg_method)