import numpy as np
import torch
from typing import Optional
import cv2
import matplotlib.pyplot as plt

def heatmap_object(img: np.ndarray, bounding_box: dict, heatmap: np.ndarray) -> np.ndarray:
    """
    This function generates the heatmaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - heatmap (np.ndarray): heatmap of the current input img

    Returns:
        - heatmap (np.ndarray): output heatmap with the current object heatmap added
    """

#     plt.imshow(heatmap)
    # Rbounding box coordinates
    hmb = heatmap
    #print(heatmap.shape)
    x1 = bounding_box['bndbox']['xmin']
    y1 = bounding_box['bndbox']['ymin']
    x2 = bounding_box['bndbox']['xmax']
    y2 = bounding_box['bndbox']['ymax']
    
    # bounding box dimensions
    box_width = x2 - x1
    box_height = y2 - y1
    

    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    shorter_side = min(box_width, box_height)
    iou_threshold = 0.3
    radius = (shorter_side / 2) * np.sqrt(1 - iou_threshold)
    
    sigma_p = radius/3
    
    
    # Update heatmap with the object's keypoint values using Gaussian kernel as mentioned in papaer "objects as points"
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            heatmap[y, x] += np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma_p ** 2))
    


#     print(heatmap.shape)
    return heatmap

def sizemap_object(img: np.ndarray, bounding_box: dict, sizemap: np.ndarray) -> np.ndarray:
    """
    This function generates the sizemaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - sizemap (np.ndarray): sizemap of the current input img

    Returns:
        - sizemap (np.ndarray): output sizemap with the current object sizemap added
    """


    

    
    x1 = bounding_box['bndbox']['xmin']
    y1 = bounding_box['bndbox']['ymin']
    x2 = bounding_box['bndbox']['xmax']
    y2 = bounding_box['bndbox']['ymax']
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    object_height = y2 - y1
    object_width = x2 - x1
    
    sizemap[center_y, center_x, 0] = object_height
    sizemap[center_y, center_x, 1] = object_width
    
    #new_sizemap = sizemap[:, :, 0:2]





    return sizemap