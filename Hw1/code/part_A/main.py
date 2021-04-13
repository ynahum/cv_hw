from my_keypoint_det import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %% Global parameters

data_path = "../../data"
sigma0 = 1
k = np.sqrt(2)
levels = np.array([-1,0,2,3,4])
th_contrast = 0.03
th_r = 12

# %% 1.1 Load chickenbroth image

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

# %% 1.3 Visualize DoG pyramid

DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
displayPyramid(DoGPyramid)

print('DoG pyramid shape: ' + str(DoGPyramid.shape))

# %% 1.4 Visualize edge suppresion - without thresholding

PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
print('Principal curvature shape: ' + str(PrincipalCurvature.shape))

plt.figure()
for i in range(len(PrincipalCurvature)):
    plt.subplot(2,2,i+1)
    plt.imshow(PrincipalCurvature[i,:,:],cmap='gray')
    plt.title('Edge suppresion at level: ' + str(i + 1))
    plt.axis('off')

plt.show()

# %% 1.4 Visualize edge suppresion - with thresholding

plt.figure()
for i in range(len(PrincipalCurvature)):
    plt.subplot(2,2,i+1)
    plt.imshow(cv2.threshold(PrincipalCurvature[i,:,:],th_r,1,cv2.THRESH_BINARY_INV)[1],cmap='gray')
    plt.title('Edge suppresion at level: ' + str(i + 1))
    plt.axis('off')

plt.show()

# %% 1.5 Print local extrema shape

locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,th_contrast, th_r)

print('Locs DoG shape: ' + str(locsDoG.shape))

# %% 1.6 Testing it

# %% Load image model_chickenbroth and test feature extractor
im = cv2.imread(f"{data_path}/model_chickenbroth.jpg")
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

runDoGdetectorWithDifferentParameters(im,sigma0,k,levels)

# %% Load image chickenbroth_04 and test feature extractor

im = cv2.imread(f"{data_path}/chickenbroth_04.jpg")
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

runDoGdetectorWithDifferentParameters(im,sigma0,k,levels)

# %% Load image OurImage and test feature extractor

im = cv2.imread(f"{data_path}/my_data/OurImage.jpeg")
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

runDoGdetectorWithDifferentParameters(im,sigma0,k,levels)
