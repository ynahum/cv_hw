from my_BRIEF import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io

# %% Global parameters
code_path = "../../code"
data_path = "../../data"

sigma0 = 1.0
k = np.sqrt(2)
levels = np.array([-1, 0, 1, 2, 3, 4])
th_contrast = 0.03
th_r = 12
patchWidth = 9
nbits = 256

# %% 2.1 make test pattern

compareX, compareY = makeTestPattern(patchWidth, nbits)

scipy.io.savemat(f"{code_path}/testPattern.mat",
                 {
                     'sigma0': sigma0,
                     'k': k,
                     'levels': levels,
                     'th_contrast': th_contrast,
                     'th_r': th_r,
                     'patchWidth': patchWidth,
                     'compareX': compareX,
                     'compareY': compareY
                 })

im = cv2.imread(f'{data_path}/model_chickenbroth.jpg')
im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
locs1, desc1 = briefLite(im_g)