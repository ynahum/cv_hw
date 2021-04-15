from my_BRIEF import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io

# %% Global parameters
code_path = "../../code"
data_path = "../../data"
sigma0 = 1
k = np.sqrt(2)
levels = np.array([-1, 0, 2, 3, 4])
th_contrast = 0.03
th_r = 12
patchWidth = 9
nbits = 256

# %% 2.1 make test pattern

compareX, compareY = makeTestPattern(patchWidth, nbits)
# print(f"{np.shape(compareX)}")
scipy.io.savemat(f"{code_path}/testPattern.mat", {'compareX': compareX, 'compareY': compareY})

im1 = cv2.imread(f'{data_path}/model_chickenbroth.jpg')
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im1_gray = im1_gray / 255
locs1, desc1 = briefLite(im1_gray)
