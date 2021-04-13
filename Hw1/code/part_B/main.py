import numpy as np
import matplotlib.pyplot as plt
import cv2
from my_keypoint_det import *
from my_BRIEF import *

patchWidth = 9
nbits = 256
compareX, compareY = makeTestPattern(patchWidth, nbits)
#print(f"{np.shape(compareX)}")
scipy.io.savemat("testPattern.mat", {'compareX':compareX, 'compareY':compareY})


im = cv2.imread('../../data/model_chickenbroth.jpg')
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

