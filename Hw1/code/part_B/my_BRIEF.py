# %%  imports for hw1 (you can add any other library as well)
from my_keypoint_det import *
import numpy as np
import scipy.io

# %% 2.1

def makeTestPattern(patchWidth, nbits):
    """
    Your code here
    """
    # uniform sampling. first imp in the article
    vec_size = np.power(patchWidth, 2)
    compareX = np.random.randint(low=0, high=vec_size, size=nbits).reshape((nbits,1))
    compareY = np.random.randint(low=0, high=vec_size, size=nbits).reshape((nbits,1))
    return compareX, compareY


# %% 2.2

def computeBrief(im, GaussianPyramid, locsDoG, k, levels, patchWidth, compareX, compareY):
    """
    Your code here
    """
    filter_half_support = int(patchWidth / 2)
    pyramid_width = GaussianPyramid.shape[2]
    pyramid_height = GaussianPyramid.shape[1]
    maskInnerPoints = (locsDoG[:, 1] >= filter_half_support) & (locsDoG[:, 1] <= pyramid_height - 1 - filter_half_support) & \
        (locsDoG[:, 0] >= filter_half_support) & (locsDoG[:, 0] <= pyramid_width - 1 - filter_half_support)
    locs = locsDoG[maskInnerPoints, :]

    m = locs.shape[0]
    n = len(compareX)
    desc = np.zeros((m, n))

    for loc_idx, loc in enumerate(locs):
        x = loc[0]
        y = loc[1]
        level = loc[2]
        actual_level_idx = np.where(levels == level)[0]
        flatSupport = GaussianPyramid[actual_level_idx, (y - filter_half_support):(y + filter_half_support + 1),
            (x - filter_half_support):(x + filter_half_support + 1)].reshape((patchWidth * patchWidth,1))
        desc[loc_idx] = (flatSupport[compareX] < flatSupport[compareY]).reshape(-1)
    return locs, desc

# %% 2.3

def briefLite(im):
    """
    Your code here
    """
    code_path = "../../code"

    paramsDict = scipy.io.loadmat(f"{code_path}/testPattern.mat")

    sigma0 = paramsDict['sigma0'][0][0]
    k = paramsDict['k'][0][0]
    levels = paramsDict['levels'][0]
    patchWidth = paramsDict['patchWidth'][0][0]
    th_contrast = paramsDict['th_contrast'][0][0]
    th_r = paramsDict['th_r'][0][0]

    locsDoG, GaussianPyramid = DoGdetector(
        im, sigma0, k, levels,
        th_contrast, th_r)

    locs, desc = computeBrief(
        im, GaussianPyramid, locsDoG, k, levels, patchWidth,
        paramsDict['compareX'], paramsDict['compareY'])

    return locs, desc

