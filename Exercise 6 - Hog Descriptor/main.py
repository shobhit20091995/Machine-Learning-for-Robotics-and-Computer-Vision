# import libraries used during this exercise
# it may be necessary to uncomment the two following pip commands
#!pip3 install opencv-python
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# the core.py file contains the plottings and other pre-defined functions
from mlrcv.core import *
from mlrcv.hog_descriptor import compute_magnitude
from mlrcv.core import CHECK, plot_compare
from mlrcv.hog_descriptor import compute_angle
from mlrcv.core import CHECK



# Load images and divide into train and validation data
(train_data, train_labels), (val_data, val_labels) = load_data('./data')

img = load_image('./data/cat.jpg')
xxx = plt.imshow(img, cmap='gray')

# imgd = cv2.imread('./data/cat.jpg')
# cv2.imshow('Image', imgd)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from mlrcv.hog_descriptor import img_to_cell
from mlrcv.core import CHECK

mag_check = compute_magnitude(CHECK)
ang_check = compute_angle(CHECK)

mag_cells = img_to_cell(mag_check)
ang_cells = img_to_cell(ang_check)

from mlrcv.hog_descriptor import create_cell_histograms, build_hog_image

hist_cells = create_cell_histograms(mag_cells, ang_cells)

hog_img = np.zeros((CHECK.shape[0], CHECK.shape[0]))
hog_img = build_hog_image(hog_img, hist_cells)

plot_compare(hog_img, 'hog')