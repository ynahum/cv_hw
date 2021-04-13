# %%  imports for hw1 (you can add any other library as well)
import numpy as np
import matplotlib.pyplot as plt
import cv2

#import my_keypoint_det

def makeTestPattern(patchWidth, nbits):
    """
    Your code here
    """
    # uniform sampling. first imp in the article
    vec_size = np.power(patchWidth, 2)
    compareX = np.random.randint(low=0, high=vec_size, size=nbits).reshape((nbits,1))
    compareY = np.random.randint(low=0, high=vec_size, size=nbits).reshape((nbits,1))
    return compareX, compareY



def computeBrief(im, GaussianPyramid, locsDoG, k, levels, patchWidth, compareX, compareY):
    """
    Your code here
    """
    print(f'GaussianPyramid.shape {GaussianPyramid.shape}')
    locs = locsDoG[:,:]
    return locs, desc


