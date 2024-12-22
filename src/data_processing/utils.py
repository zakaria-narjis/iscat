
import os
import nd2  
from tifffile import imwrite 

class Utils_legacy:
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
    
    @staticmethod
    def process_nd2_file(file_path,indices):
    # Open the .nd2 file
        with nd2.ND2File(file_path) as f:
            array = f.asarray()  # Load the entire stack as a NumPy array
            num_frames = array.shape[0]  # Number of frames
            # Save each specified frame as a 16-bit TIFF
            for idx in indices:
                if 0 <= idx < num_frames:
                    frame = array[idx]
                    # Generate the output path
                    output_path = os.path.join(
                        os.path.dirname(file_path), 
                        f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{idx:03d}.tiff"
                    )
                    # Save the frame as a 16-bit TIFF
                    imwrite(output_path, frame, dtype='uint16')
                    print(f"Saved: {output_path}")

class Utils:
    @staticmethod
    def get_data_paths(root_path, mode="Brightfield", image_indices=[0,100,200]):
        """
        Extract paths to .nd2 files and corresponding TIFF files from the specified mode folder.

        Args:
            root_path (str): The root directory to search.
            mode (str): The folder name to focus on (default is 'Brightfield').

        Returns:
            tuple: Two lists - list of .nd2 file paths and list of tuples with corresponding TIFF file paths.
        """

        zimages_files = []
        target_files = []


        for dirpath, dirnames, filenames in os.walk(root_path):
            if os.path.basename(dirpath) == mode:
                 
                zimages_files.append([os.path.join(dirpath, f"frame_{idx}.tiff") for idx in image_indices])
                if  not all(os.path.exists(path) for path in zimages_files[-1]): 
                    nd2_path = None
                    for file in filenames:
                        if file.endswith('.nd2'): 
                            nd2_path = os.path.join(dirpath, file)
                            Utils.process_nd2_file(nd2_path,dirpath,image_indices)
                            break
                    if nd2_path is None:
                        raise Exception("No .nd2 file found")
                # Generate TIFF file paths dynamically based on the prefix
                cy5_path = os.path.join(dirpath, f'Captured Cy5.tif')
                fitc_path = os.path.join(dirpath, f'Captured FITC.tif')
                tritc_path = os.path.join(dirpath, f'Captured TRITC.tif')
                target_files.append((cy5_path, fitc_path, tritc_path))
                
                # Ensure all three TIFF files exist
                assert all(os.path.exists(path) for path in [cy5_path, fitc_path, tritc_path])
                assert all(os.path.exists(path) for path in zimages_files[-1])                 
                
        return zimages_files, target_files
    
    @staticmethod
    def process_nd2_file(nd2_file,dirpath,indices):
    # Open the .nd2 file
        with nd2.ND2File(nd2_file) as f:
            array = f.asarray()  # Load the entire stack as a NumPy array
            num_frames = array.shape[0]  # Number of frames
            # Save each specified frame as a 16-bit TIFF
            for idx in indices:
                assert 0 <= idx < num_frames
                frame = array[idx]
                # Generate the output path
                output_path = os.path.join(
                    dirpath, 
                    f"frame_{idx}.tiff"
                )
                # Save the frame as a 16-bit TIFF
                imwrite(output_path, frame, dtype='uint16')
                print(f"Saved: {output_path}")