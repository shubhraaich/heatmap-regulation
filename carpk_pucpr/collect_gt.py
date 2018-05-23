""" Connected component analysis source:
https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
"""

from skimage import measure
import numpy as np
import cv2

def get_gt_counts(im) :
    """ return simple counts from the RGB (8-bit) image, numpy ndarray im """
    im_bw = np.logical_and(im[:,:,1]<50, im[:,:,2]<50);
    im_bw = np.logical_and(im[:,:,0]>200, im_bw);
    labels = measure.label(im_bw, neighbors=8, background=False);
    return len(np.unique(labels)) - 1;
