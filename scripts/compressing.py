import os
import zipfile
from tqdm import tqdm
import argparse

def compress_chip_folders(chip_path, compress_type="Brightfield", output_zip_path=None):
    """
    Compresses specified folders ("Brightfield" or "Laser") within metasurface folders under a chip path into a structured zip file.

    Args:
        chip_path (str): Path to the chip folder containing metasurface folders.
        compress_type (str): Folder to compress within each metasurface folder ("Brightfield" or "Laser").
        output_zip_path (str, optional): Path to save the output zip file. Defaults to chip_X.zip in the same directory.

    Returns:
        str: Path to the created zip file.
    """
    if not os.path.isdir(chip_path):
        raise ValueError(f"The specified chip path does not exist or is not a directory: {chip_path}")

    # Default output zip path
    if output_zip_path is None:
        output_zip_path = os.path.join(os.path.dirname(chip_path), f"{os.path.basename(chip_path)}.zip")

    metasurface_folders = []
    for root, dirs, _ in os.walk(chip_path):
        for dir_name in dirs:
            if dir_name.startswith("Metasurface"):
                metasurface_folders.append(os.path.join(root, dir_name))

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for metasurface_path in tqdm(metasurface_folders, desc="Processing metasurface folders"):
            target_folder_path = os.path.join(metasurface_path, compress_type)

            if os.path.isdir(target_folder_path):
                for folder_root, _, folder_files in os.walk(target_folder_path):
                    for file in folder_files:
                        file_path = os.path.join(folder_root, file)
                        # Generate a structured archive path
                        arcname = os.path.relpath(file_path, chip_path)
                        zipf.write(file_path, arcname)

    return output_zip_path

def main():
    parser = argparse.ArgumentParser(description="Compress metasurface folders into a structured zip file.")
    parser.add_argument("chip_path", type=str, help="Path to the chip folder containing metasurface folders.")
    parser.add_argument("compress_type", type=str, choices=["Brightfield", "Laser"], help="Folder type to compress (Brightfield or Laser).")
    parser.add_argument("--output_zip_path", type=str, default=None, help="Path to save the output zip file. Defaults to chip_X.zip in the same directory.")

    args = parser.parse_args()

    output_zip_path = compress_chip_folders(args.chip_path, args.compress_type, args.output_zip_path)
    print(f"Compressed file saved at: {output_zip_path}")

if __name__ == "__main__":
    main()
