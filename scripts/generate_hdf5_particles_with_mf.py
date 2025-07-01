import os
import h5py
import numpy as np
import pandas as pd
from nd2 import ND2File
from pathlib import Path
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processing.utils import subtract_median_background_chunked
import argparse


def get_nd2_and_csv_paths(base_path, option):
    mask_names = ["Captured Cy5.csv", "Captured FITC.csv", "Captured TRITC.csv"]
    class_idx = {
        "Captured Cy5.csv": 0,
        "Captured FITC.csv": 1,
        "Captured TRITC.csv": 2,
    }
    class_idx_2024_11_29 = {
        "Captured Cy5.csv": 3,
        "Captured FITC.csv": 0,
        "Captured TRITC.csv": 1,
    }

    file_pairs = []
    for root, _, files in os.walk(base_path):
        if "Metasurface" in Path(root).parts:
            target_folder = os.path.join(root, option)
            if os.path.isdir(target_folder):
                for file in os.listdir(target_folder):
                    if file.endswith(".nd2"):
                        nd2_path = os.path.join(target_folder, file)
                        csv_paths = []
                        for mask_name in mask_names:
                            csv_path = os.path.join(target_folder, mask_name)
                            if os.path.exists(csv_path):
                                if "2024_11_29" in Path(root).parts:
                                    csv_paths.append((csv_path, class_idx_2024_11_29[mask_name]))
                                else:
                                    csv_paths.append((csv_path, class_idx[mask_name]))
                        if csv_paths:
                            file_pairs.append((nd2_path, csv_paths))
    return file_pairs


def pad_or_crop_to_16x16(region):
    """
    Adjust the region to shape (Z, 16, 16) by padding or cropping on H and W.
    """
    z, h, w = region.shape
    target_size = 16

    # Crop if larger than target
    if h > target_size:
        start_h = (h - target_size) // 2
        region = region[:, start_h:start_h+target_size, :]
    if w > target_size:
        start_w = (w - target_size) // 2
        region = region[:, :, start_w:start_w+target_size]

    # Pad if smaller
    pad_h = max(0, target_size - region.shape[1])
    pad_w = max(0, target_size - region.shape[2])

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    region = np.pad(
        region,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge"
    )

    return region


def extract_particle_region(image_stack, bbox, offset=0):
    xmin, xmax, ymin, ymax = bbox
    xmin = max(xmin - offset, 0)
    xmax = min(xmax + offset, image_stack.shape[2])
    ymin = max(ymin - offset, 0)
    ymax = min(ymax + offset, image_stack.shape[1])
    return image_stack[:, ymin:ymax, xmin:xmax]


def process_3d_particles(
    file_pairs,
    output_path,
    offset=0,
    kernel_size=31,
    chunk_size=16,
    device="cpu"
):
    data_list = []
    labels_list = []

    with h5py.File(output_path, "w") as hf:
        for nd2_path, csv_paths in file_pairs:
            print(f"Processing {nd2_path}")
            with ND2File(nd2_path) as nd2:
                image = nd2.asarray()  # (Z, H, W)
                print("Applying median filter")
                image = subtract_median_background_chunked(
                    image,
                    kernel_size=kernel_size,
                    chunk_size=chunk_size,
                    device=device,
                    method='chunked'
                )

                for csv_path, class_idx in csv_paths:
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        bbox = (
                            int(row["xMin"]),
                            int(row["xMax"]),
                            int(row["yMin"]),
                            int(row["yMax"])
                        )
                        region = extract_particle_region(image, bbox, offset=offset)
                        region = pad_or_crop_to_16x16(region)
                        data_list.append(region)
                        labels_list.append(class_idx)

        data_array = np.stack(data_list)  # (N, Z, 16, 16)
        labels_array = np.array(labels_list)

        hf.create_dataset("data", data=data_array)
        hf.create_dataset("labels", data=labels_array)
        hf.attrs["description"] = "3D particle crops (Z, 16, 16) after median filter"
        hf.attrs["median_filter_kernel"] = kernel_size
        hf.attrs["offset"] = offset
        hf.attrs["class_info"] = (
            "Class 0: Cy5 (80nm), Class 1: FITC (300nm), Class 2: TRITC (1300nm), Class 3: (600nm)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D particle crops with MF")
    parser.add_argument("--datatype", type=str, default="Brightfield", help="Type of data to process (e.g., Brightfield, Laser)")
    parser.add_argument("--output_path", type=str, default="particles_3d_mf.hdf5")
    parser.add_argument("--offset", type=int, default=2, help="Offset added to bounding box")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--kernel_size", type=int, default=27, help="Median filter kernel size")
    parser.add_argument("--chunk_size", type=int, default=4, help="Chunk size for processing")

    args = parser.parse_args()

    data_paths = [
        os.path.join("data", "2024_11_11", "Metasurface", "Chip_02"),
        os.path.join("data", "2024_11_12", "Metasurface", "Chip_01"),
        os.path.join("data", "2024_11_29", "Metasurface", "Chip_02"),
    ]

    all_file_pairs = []
    for path in data_paths:
        all_file_pairs.extend(get_nd2_and_csv_paths(path, args.datatype))

    process_3d_particles(
        all_file_pairs,
        args.output_path,
        offset=args.offset,
        kernel_size=args.kernel_size,
        chunk_size=args.chunk_size,
        device=args.device,
    )
