from my_keypoint_det import *
from my_BRIEF import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io

# %% Global parameters

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
scipy.io.savemat(f"{data_path}/my_data/testPattern.mat", {'compareX': compareX, 'compareY': compareY})

im = cv2.imread(f"{data_path}/model_chickenbroth.jpg")
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

# %% 1.2 Preprocess image

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

# %% 1.2 Process gaussian pyramid and display

GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
displayPyramid(GaussianPyramid)

print('Gaussian pyramid shape: ' + str(GaussianPyramid.shape))
