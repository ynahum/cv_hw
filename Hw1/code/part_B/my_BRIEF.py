# %%  imports for hw1 (you can add any other library as well)
from my_keypoint_det import *
import numpy as np
import scipy.io

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
    filter_half_support = int(patchWidth / 2)
    pyramid_width = GaussianPyramid.shape[2]
    pyramid_height = GaussianPyramid.shape[1]
    maskInnerPoints = (locsDoG[:, 1] >= filter_half_support & (locsDoG[:, 1] < pyramid_width - filter_half_support)) & \
        (locsDoG[:, 2] >= filter_half_support & (locsDoG[:, 2] < pyramid_height - filter_half_support))
    locs = locsDoG[maskInnerPoints, :]
    m = locs.shape[0]
    n = len(compareX)
    desc = np.zeros((m, n))

    for row_idx, row in enumerate(locs):
        x = row[0]
        y = row[1]
        level = row[2]
        actual_level_idx = np.where(levels == level)[0]
        flatSupport = GaussianPyramid[actual_level_idx, (y - filter_half_support):(y + filter_half_support + 1),
            (x - filter_half_support):(x + filter_half_support + 1)].reshape(-1)
        positive_locations = (flatSupport[compareX] < flatSupport[compareY])
        desc[row_idx] = positive_locations.reshape((1,n))
    return locs, desc

def briefLite(im):
    """
    Your code here
    """
    sigma0 = 1
    k = np.sqrt(2)
    patchWidth = 9
    levels = np.array([-1, 0, 1, 2, 3, 4])  # np.array([1, 2, 3, 4, 6])
    code_path = "../../code"
    compareDict = scipy.io.loadmat(f"{code_path}/testPattern.mat")
    locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, th_contrast=0.03, th_r=12)
    locs, desc = computeBrief(im, GaussianPyramid, locsDoG, k, levels, patchWidth, compareDict['compareX'], compareDict['compareY'])
    return locs, desc
