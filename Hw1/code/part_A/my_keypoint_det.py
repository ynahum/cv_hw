# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:13:23 2021

@author: daniel
"""

# %%  imports for hw1 (you can add any other library as well)
import numpy as np
import matplotlib.pyplot as plt
import cv2


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
    plt.show()
    

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
        epsilon = 10 ** (-7)
        # we add epsilon to avoid division by zero when hessian is not invertible (det = 0)
        R = np.abs(np.divide(np.power(TrH,2),DetH + epsilon))
        PrincipalCurvature.append(R)
    PrincipalCurvature = np.stack(PrincipalCurvature)
    PrincipalCurvature = np.abs(PrincipalCurvature)
    return PrincipalCurvature


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
    mask = (np.abs(DoGPyramid) > th_contrast) & (PrincipalCurvature < th_r)
    maskInd = np.where(mask)
    lvlMax = len(mask)-1
    rows = mask.shape[1]; cols = mask.shape[2]
    nSize = 1 # Neighboorhod size
    locsDoG = []
    for lvlInd, r, c in zip(maskInd[0],maskInd[1],maskInd[2]):
        lvlPrev = np.max([0,lvlInd-nSize]); lvlNext = np.min([lvlMax + 1,lvlInd + nSize + 1]);
        rCurMin = np.max([0, r-nSize]); rCurMax = np.min([rows, r + nSize + 1]);
        cCurMin = np.max([0, c-nSize]); cCurMax = np.min([cols, c + nSize + 1]);
        curNeighborhood =  DoGPyramid[lvlPrev:lvlNext,rCurMin:rCurMax,cCurMin:cCurMax]
        if DoGPyramid[lvlInd,r,c] == np.min(curNeighborhood) or  DoGPyramid[lvlInd,r,c] == np.max(curNeighborhood):
            locsDoG.append(np.array([c,r, lvlInd]))
    locsDoG = np.array(locsDoG)
    return locsDoG


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
        plt.subplot(5,1,i+1)
        locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, th_contrast[i], th_r[i])
        im_with_detector = np.copy(im)
        im_with_detector[np.int64(locsDoG[:,1]),np.int64(locsDoG[:,0])] = 1
        plt.imshow(im_with_detector,cmap = 'gray')
        plt.title('$Th_r = $' + str(th_r[i]) + ' and $ Th_{contrast} = $' + str(th_contrast[i]))
        plt.axis('off')
    plt.show()








