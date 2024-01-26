import numpy as np
import cv2
from mlrcv.core import *
from typing import Optional

def compute_magnitude(img: np.ndarray) -> np.ndarray:
    """
    This function computes the magnitudes for each pixel in an image:

    Args:
        - img (np.ndarray): image to compute the magnitude

    Returns:
        - magnitude (np.ndarray): magnitude computed for each pixel
    """

    magnitude = np.zeros((img.shape[0], img.shape[1]))
    

    # Compute the gradients and magnitude for each pixel
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            gx = img[i][j+1] - img[i][j-1]
            gy = img[i+1][j] - img[i-1][j]
            magnitude[i][j] = np.sqrt(gx**2 + gy**2)

    return magnitude

def compute_angle(img: np.ndarray) -> np.ndarray:
    """
    This function computes the angles for each pixel in an image:

    Args:
        - img (np.ndarray): image to compute the magnitude

    Returns:
        - angle (np.ndarray): angles computed for each pixel
    """

    angle = np.zeros((img.shape[0], img.shape[1]))
    magnitude = np.zeros((img.shape[0], img.shape[1]))

    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            gx = img[i][j+1] - img[i][j-1]
            gy = img[i+1][j] - img[i-1][j]
            magnitude[i][j] = np.sqrt(gx**2 + gy**2)
            angle[i][j] = np.arctan2(gy, gx)
              

   
    # used to convert to degrees 
    angle = np.degrees(angle)
    # taking absolute value
    angle = np.abs(angle)  

        

    

    return angle

def create_cell_histograms(mag_cells: np.ndarray, ang_cells: np.ndarray, bin_size: Optional[int] = 9) -> np.ndarray:
    """
    This function computes the histograms for each cell in the image already divided into 8x8 cells:

    Args:
        - mag_cells (np.ndarray): magnitude values divided into cells
        - ang_cells (np.ndarray): angles values divided into cells
        - bin_size (int): number of bins on the histogram

    Returns:
        - bins (np.ndarray): Histogram calculated in each 8x8 cell in the image
    """

    bins = np.zeros((mag_cells.shape[0], mag_cells.shape[1], bin_size))
    step_size = 180 / bin_size
    
    for k in range(mag_cells.shape[0]):
        for l in range(mag_cells.shape[1]):
            for i in range(mag_cells.shape[2]):
                for j in range(mag_cells.shape[3]):
                   
                    bin = int(ang_cells[k,l,i,j] / step_size) % bin_size
                    
                    
                    bins[k,l,bin] += mag_cells[k,l,i,j]

   

    return bins

def dataloader(data_files: np.ndarray) -> np.ndarray:
    """
    This function load the images listed in data_files and compute the magnitudes and angles
    to calculate the histograms to extract the hog features

    Args:
        - data_files (np.ndarray): list of image files to be loaded and generate the hog features

    Returns:
        - hog_feats (np.ndarray): hog features for all the images listed in data_files
    """

    data_feats = []
    for fname in data_files:
        img = load_image(fname)

        ###########################################
        # Implement here you function:
        # - hist_cells should be output histogram from create_cell_histograms function
        ###########################################
        
        
        mag = compute_magnitude(img)
        ang = compute_angle(img)
        mag_cells = img_to_cell(mag)
        ang_cells = img_to_cell(ang)
        hist_cells = create_cell_histograms(mag_cells, ang_cells)

        pass

        ###########################################

        hog_img = np.zeros((img.shape[0], img.shape[1]))
        hog_img = build_hog_image(hog_img, hist_cells)
        save_image(hog_img, fname)
        feats = bins_to_feats(hist_cells)

        data_feats.append(feats)

    hog_feats = np.vstack(data_feats)

    return hog_feats
