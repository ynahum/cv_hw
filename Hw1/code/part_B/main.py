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


# %% 2.4 descriptors matching
from testMatch import *

testMatch(f'{data_path}/chickenbroth_01.jpg', f'{data_path}/chickenbroth_04.jpg')
#testMatch(f'{data_path}/chickenbroth_01.jpg', f'{data_path}/chickenbroth_03.jpg')

testMatch(f'{data_path}/incline_L.png', f'{data_path}/incline_R.png')

pf_scan_scaled_path = f'{data_path}/pf_scan_scaled.jpg'
testMatch(pf_scan_scaled_path, f'{data_path}/pf_desk.jpg')
testMatch(pf_scan_scaled_path, f'{data_path}/pf_floor.jpg')
testMatch(pf_scan_scaled_path, f'{data_path}/pf_floor_rot.jpg')
testMatch(pf_scan_scaled_path, f'{data_path}/pf_pile.jpg')
testMatch(pf_scan_scaled_path, f'{data_path}/pf_stand.jpg')

#self matching
testMatch(f'{data_path}/chickenbroth_01.jpg', f'{data_path}/chickenbroth_01.jpg')
