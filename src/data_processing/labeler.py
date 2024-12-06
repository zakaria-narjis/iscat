from PIL import Image
import numpy as np
import os
import pandas as pd
import imagej
import scyjava as sj

class Labeler:

    def __init__(self,):
        self.ij = imagej.init(ij_dir_or_version_or_endpoint='D:\Fiji.app',mode=imagej.Mode.HEADLESS)
        print(f"ImageJ2 version: {self.ij.getVersion()}")
        pass

    def create_labels_mask(self, canvas_shape, particles_positions):
        """
            canvas_shape: tuple of image shape (height, width)
            particles_positions: list of pandas dataframe of bounding box of particles in each image
            return: numpy array of labels (mask)
        """
        # Preallocate canvas
        label_canvas = np.zeros(canvas_shape, dtype=np.uint8)

        # Precompute y, x coordinates
        y, x = np.ogrid[:canvas_shape[0], :canvas_shape[1]]

        for particle_data in particles_positions:
            for _, row in particle_data.iterrows():
                # Compute ellipse parameters
                center_x = int((row['xMin'] + row['xMax']) / 2)
                center_y = int((row['yMin'] + row['yMax']) / 2)
                axes_x = int((row['xMax'] - row['xMin']) / 2)
                axes_y = int((row['yMax'] - row['yMin']) / 2)

                # Vectorized ellipse mask creation
                mask = ((x - center_x) / axes_x)**2 + ((y - center_y) / axes_y)**2 <= 1

                # Efficient overlap and labeling
                label_canvas[mask & (label_canvas == 1)] = 2
                label_canvas[mask & (label_canvas == 0)] = 1

        return label_canvas 

    def label(self,fluo_images_paths,seg_args,segmentation_method="comdet"):
        """
        fluo_images_paths: list of fluorecense images paths
        seg_args: list of segmentation arguments
        segmentation_method: method to segment the particles
        return: numpy array of labels (mask)
        """
        if segmentation_method == "comdet":
            particles_positions = []
            for image_path,args in zip(fluo_images_paths,seg_args):
                particles_position_path = image_path.replace(".tif",".csv")
                if not os.path.exists(particles_position_path):  
                    image = self.ij.IJ.openImage(image_path)            
                    self.ij.py.run_plugin(plugin="Detect Particles",args=args,imp=image)
                    table = self.ij.ResultsTable.getResultsTable()
                    Table = sj.jimport('org.scijava.table.Table')
                    results = self.ij.convert().convert(table, Table)
                    results = self.ij.py.from_java(results)        
                    results.to_csv(particles_position_path)
                    particles_positions.append(results)
                else:
                    particles_positions.append(pd.read_csv(particles_position_path))
            canvas_shape = np.array(Image.open(fluo_images_paths[0])).shape
            mask = self.create_labels_mask(canvas_shape, particles_positions)
        else:
            raise ValueError("Invalid segmentation method")
        return mask

