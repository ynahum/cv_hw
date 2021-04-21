# %%  imports for hw1 (you can add any other library as well)
from my_keypoint_det import *
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist


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
        desc[loc_idx] = (flatSupport[compareX] < flatSupport[compareY]).reshape((n,))
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


# %% 2.4 descriptors matching

def briefMatch(desc1, desc2, ratio=0.6):
    #     performs the descriptor matching
    #     inputs  : desc1 , desc2 - m1 x n and m2 x n matrices. m1 and m2 are the number of keypoints in image 1 and 2.
    #                               n is the number of bits in the brief
    #               ratio         - ratio used for testing whether two descriptors should be matched.
    #     outputs : matches       - p x 2 matrix. where the first column are indices
    #                                         into desc1 and the second column are indices into desc2
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1,ix2), axis=-1)
    return matches
