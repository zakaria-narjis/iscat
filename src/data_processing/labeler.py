from PIL import Image
import numpy as np
import os
import pandas as pd
import imagej
import scyjava as sj
import cv2

class Labeler:

    def __init__(self,method="comdet"):
        if method == "comdet":
            self.ij = imagej.init(ij_dir_or_version_or_endpoint='D:\Fiji.app',mode=imagej.Mode.HEADLESS)
            print(f"ImageJ2 version: {self.ij.getVersion()}")

    def create_labels_mask(self, canvas_shape, particles_positions_df):
        """
            canvas_shape: tuple of image shape (height, width)
            particles_positions: list of pandas dataframe of bounding box of particles in each image
            return: numpy array of labels (mask)
        """
        # Preallocate canvas
        label_canvas = np.zeros(canvas_shape, dtype=np.uint8)

        # Precompute y, x coordinates
        y, x = np.ogrid[:canvas_shape[0], :canvas_shape[1]]

        for _, row in particles_positions_df.iterrows():
            # Compute ellipse parameters
            center_x = int((row['xMin'] + row['xMax']) / 2)
            center_y = int((row['yMin'] + row['yMax']) / 2)
            axes_x = int((row['xMax'] - row['xMin']) / 2)
            axes_y = int((row['yMax'] - row['yMin']) / 2)

            # Vectorized ellipse mask creation
            ellipse_mask = ((x - center_x) / axes_x)**2 + ((y - center_y) / axes_y)**2 <= 1

            # Efficient overlap and labeling
            label_canvas[ellipse_mask] = 1

        return label_canvas 

    def generate_labels(self,fluo_images_paths,seg_args=None,segmentation_method="comdet"):
        """
        fluo_images_paths: list of fluorecense images paths
        seg_args: list of segmentation arguments
        segmentation_method: method to segment the particles

        """
        if segmentation_method == "comdet":
            particles_positions = []
            canvas_shape = np.array(Image.open(fluo_images_paths[0])).shape
            
            for fluo_image_path, args in zip(fluo_images_paths, seg_args):
                particles_position_path = fluo_image_path.replace(".tif",".csv")
                if not os.path.exists(particles_position_path):  
                    image = self.ij.IJ.openImage(fluo_image_path)            
                    self.ij.py.run_plugin(plugin="Detect Particles",args=args,imp=image)
                    table = self.ij.ResultsTable.getResultsTable()
                    Table = sj.jimport('org.scijava.table.Table')
                    results = self.ij.convert().convert(table, Table)
                    results = self.ij.py.from_java(results)        
                    results.to_csv(particles_position_path)
                    particles_positions.append(results)
                else:
                    particles_positions_df = pd.read_csv(particles_position_path)
                    mask = self.create_labels_mask(canvas_shape, particles_positions_df)
                    
                    mask_path = fluo_image_path.replace(".tif","_mask.npy")
                np.save(mask_path,mask)
        elif segmentation_method == "kmeans":
            for fluo_image_path in fluo_images_paths:
                mask_path = fluo_image_path.replace(".tif", "_mask_kmeans.npy")
                # Load the fluorescence image
                image = cv2.imread(fluo_image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(f"Image not found at {fluo_image_path}")

                # Normalize 16-bit image to 8-bit for KMeans processing
                image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Perform KMeans segmentation
                def kmeans_color_quantization(image, clusters=2, rounds=1):
                    h, w = image.shape[:2]
                    samples = np.zeros([h * w, 1], dtype=np.float32)
                    count = 0

                    for x in range(h):
                        for y in range(w):
                            samples[count] = image[x, y]
                            count += 1

                    compactness, labels, centers = cv2.kmeans(
                        samples,
                        clusters,
                        None,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                        rounds,
                        cv2.KMEANS_RANDOM_CENTERS
                    )

                    centers = np.uint16(centers)
                    res = centers[labels.flatten()]
                    return res.reshape((image.shape))

                kmeans_result = kmeans_color_quantization(image_normalized, clusters=2)

                # Apply Otsu's thresholding to get binary mask
                kmeans_normalized = cv2.normalize(kmeans_result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                thresh = cv2.threshold(kmeans_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # Save mask as numpy file
                np.save(mask_path, thresh)
        else:
            raise ValueError("Invalid segmentation method")


