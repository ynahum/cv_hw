# %%  imports for hw1 (you can add any other library as well)
import numpy as np
import matplotlib.pyplot as plt
import cv2
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


patchWidth = 9
nbits = 256
compareX, compareY = makeTestPattern(patchWidth, nbits)
#print(f"{np.shape(compareX)}")
scipy.io.savemat("testPattern.mat", {'compareX':compareX, 'compareY':compareY})



