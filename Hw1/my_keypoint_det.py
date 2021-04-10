# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:13:23 2021

@author: daniel
"""

# %%  imports for hw1 (you can add any other library as well)
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %% Global parameters

sigma0 = 1
k = np.sqrt(2)
levels = np.array([-1,0,2,3,4])
th_contrast = 0.03
th_r = 12

# %% 1.1 Load chickenbroth image

im = cv2.imread('data/model_chickenbroth.jpg')
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')

# %% 1.2 Gaussian pyramid

def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor( 3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im,(size,size),sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)

# %% 1.2 Visualize gaussian pyramid

def displayPyramid(pyramid):
    plt.figure(figsize=(16,5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')
    
# %% 1.2 Preprocess image

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

# %% 1.2 Process gaussian pyramid and display

GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
displayPyramid(GaussianPyramid)

print('Gaussian pyramid shape: ' + str(GaussianPyramid.shape))

# %% 1.3 The DoG Pyramid

def createDoGPyramid(GaussianPyramid, levels):
    # Produces DoG Pyramid
    # inputs
    # GaussianPyramid - A matrix of grayscale images of size
    # (len(levels), shape(im))
    # levels - the levels of the pyramid where the blur at each level is
    # outputs
    # DoGPyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    # created by differencing the Gaussian Pyramid input
    # DogLevels - the levels of the pyramid where the blur at each level corresponds
    # to the DoG scale
    """
    Your code here
    """
    DoGPyramid = []
    for i in range(len(levels)-1):
        DoGPyramid.append(GaussianPyramid[i,:,:] - GaussianPyramid[i+1,:,:])
    DoGLevels = levels[1:]
    DoGPyramid = np.stack(DoGPyramid)
    return DoGPyramid, DoGLevels

# %% 1.3 Visualize DoG pyramid

DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
displayPyramid(DoGPyramid)

print('DoG pyramid shape: ' + str(DoGPyramid.shape))

# %% 1.4 Edge suppression

def computePrincipalCurvature(DoGPyramid):
    # Edge Suppression
    # Takes in DoGPyramid generated in createDoGPyramid and returns
    # PrincipalCurvature,a matrix of the same size where each point contains the
    # curvature ratio R for the corre-sponding point in the DoG pyramid
    #
    # INPUTS
    # DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    # OUTPUTS
    # PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    # point contains the curvature ratio R for the
    # corresponding point in the DoG pyramid
    """
    Your code here
    """
    PrincipalCurvature = []
    for i in range(len(DoGPyramid)):
        Dxx = cv2.Sobel(DoGPyramid[i,:,:],cv2.CV_64F,2,0)
        Dyy = cv2.Sobel(DoGPyramid[i,:,:],cv2.CV_64F,0,2)
        Dxy = cv2.Sobel(DoGPyramid[i,:,:],cv2.CV_64F,1,1)
        TrH = Dxx + Dyy
        DetH = np.multiply(Dxx,Dyy) - np.multiply(Dxy,Dxy)
        R = np.divide(np.power(TrH,2),DetH)
        PrincipalCurvature.append(R)
    PrincipalCurvature = np.stack(PrincipalCurvature)
    return PrincipalCurvature

# %% 1.4 Visualize edge suppresion - without thresholding

PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
print('Principal curvature shape: ' + str(PrincipalCurvature.shape))

plt.figure()
for i in range(len(PrincipalCurvature)):
    plt.subplot(2,2,i+1)
    plt.imshow(PrincipalCurvature[i,:,:],cmap='gray')
    plt.title('Edge suppresion at level: ' + str(i + 1))
    plt.axis('off')
    
# %% 1.4 Visualize edge suppresion - with thresholding

plt.figure()
for i in range(len(PrincipalCurvature)):
    plt.subplot(2,2,i+1)
    plt.imshow(cv2.threshold(PrincipalCurvature[i,:,:],th_r,1,cv2.THRESH_BINARY_INV)[1],cmap='gray')
    plt.title('Edge suppresion at level: ' + str(i + 1))
    plt.axis('off')
    
# %% 1.5 Get local extrema

def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,th_contrast, th_r):
    # Returns local extrema points in both scale and space using the DoGPyramid
    # INPUTS
    # DoGPyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    # DoGlevels - The levels of the pyramid where the blur at each level is
    # outputs
    # PrincipalCurvature - size (len(levels) - 1, imH, imW) matrix contains the
    # curvature ratio R
    # th_contrast - remove any point that is a local extremum but does not have a
    # DoG response magnitude above this threshold
    # th_r - remove any edge-like points that have too large a principal
    # curvature ratio
    # OUTPUTS
    # locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in
    # scale and space, and also satisfies the two thresholds.
    """
    Your code here
    """
    locsDoGMatrix = 1
    for i in range(len(DoGLevels)):
        DoGPyramidLevelTh = cv2.threshold(np.abs(DoGPyramid[i,:,:]),th_contrast,1,cv2.THRESH_BINARY)[1]
        PrincipalCurvatureLevelTh = cv2.threshold(PrincipalCurvature[i,:,:],th_r,1,cv2.THRESH_TOZERO_INV)[1]
        localExtremaMatrix = np.multiply(DoGPyramidLevelTh,PrincipalCurvatureLevelTh)
        locsDoGMatrix = np.multiply(locsDoGMatrix,localExtremaMatrix)
    locsDoG = np.where(locsDoGMatrix != 0)
    locsDoGValue = locsDoGMatrix[locsDoG]
    locsDoG = np.transpose(np.array([locsDoG[1],locsDoG[0], locsDoGValue]))
    return locsDoG

# %% 1.5 Print local extrema shape

locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,th_contrast, th_r)

print('Locs DoG shape: ' + str(locsDoG.shape))

# %% 1.6 Putting it Together

def DoGdetector(im, sigma0, k, levels, th_contrast=0.03, th_r=12):
    # Putting it all together
    # Inputs Description
    # --------------------------------------------------------------------------
    # im Grayscale image with range [0,1].
    # sigma0 Scale of the 0th image pyramid.
    # k Pyramid Factor. Suggest sqrt(2).
    # levels Levels of pyramid to construct. Suggest -1:4.
    # th_contrast DoG contrast threshold. Suggest 0.03.
    # th_r Principal Ratio threshold. Suggest 12.
    # Outputs Description
    # --------------------------------------------------------------------------
    # locsDoG N x 3 matrix where the DoG pyramid achieves a local extrema
    # in both scale and space, and satisfies the two thresholds.
    # gauss_pyramid A matrix of grayscale images of size (len(levels),imH,imW)
    """
    Your code here
    """
    GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
    PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
    locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,th_contrast, th_r)
    return locsDoG, GaussianPyramid

# %% 1.6 Testing it

def runDoGdetectorWithDifferentParameters(im,sigma0,k,levels):
    th_r=[12,12,12,1,10000]
    th_contrast=[0.03,0.015,0.005,0.03,0.03]
    plt.figure()
    for i in range(len(th_r)):
        plt.subplot(1,5,i+1)
        locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, th_contrast[i], th_r[i])
        im_with_detector = np.copy(im)
        im_with_detector[np.int64(locsDoG[:,1]),np.int64(locsDoG[:,0])] = 1
        plt.imshow(im_with_detector,cmap = 'gray')
        plt.title('$Th_r = $' + str(th_r[i]) + ' and $ Th_{contrast} = $' + str(th_contrast[i]))
        plt.axis('off')

# %% Load image model_chickenbroth and test feature extractor
im = cv2.imread('data/model_chickenbroth.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

runDoGdetectorWithDifferentParameters(im,sigma0,k,levels)

# %% Load image chickenbroth_04 and test feature extractor

im = cv2.imread('data/chickenbroth_04.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

runDoGdetectorWithDifferentParameters(im,sigma0,k,levels)

# %% Load image OurImage and test feature extractor

im = cv2.imread('ImageResults/OurImage.jpeg')
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = im/255

runDoGdetectorWithDifferentParameters(im,sigma0,k,levels)







