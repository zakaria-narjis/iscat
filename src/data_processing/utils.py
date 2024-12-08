
import os

class Utils:
    @staticmethod
    def get_data_paths(root_path, mode="Brightfield"):
        """
        Extract paths to .nd2 files and corresponding TIFF files from the specified mode folder.

        Args:
            root_path (str): The root directory to search.
            mode (str): The folder name to focus on (default is 'Brightfield').

        Returns:
            tuple: Two lists - list of .nd2 file paths and list of tuples with corresponding TIFF file paths.
        """
        nd2_files = []
        tiff_files = []
        
        for dirpath, dirnames, filenames in os.walk(root_path):
            if os.path.basename(dirpath) == mode:  # Focus on the specified mode folder
                for file in filenames:
                    if file.endswith('.nd2'):  # Check for .nd2 files
                        nd2_path = os.path.join(dirpath, file)
                        
                        # Generate TIFF file paths dynamically based on the prefix
                        cy5_path = os.path.join(dirpath, f'Captured Cy5.tif')
                        fitc_path = os.path.join(dirpath, f'Captured FITC.tif')
                        tritc_path = os.path.join(dirpath, f'Captured TRITC.tif')
                        
                        # Ensure all three TIFF files exist
                        if all(os.path.exists(path) for path in [cy5_path, fitc_path, tritc_path]):
                            nd2_files.append(nd2_path)
                            tiff_files.append((cy5_path, fitc_path, tritc_path))
        
        return nd2_files, tiff_files
