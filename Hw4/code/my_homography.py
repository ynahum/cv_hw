# %% Imports

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.interpolate

# %% Functions

# Our functions

def d2ToHomoginic(p):
    pHom = np.concatenate((p,np.ones((1,p.shape[1]))))
    return pHom

def getTransformation(p,H2to1):
    pTransformed = H2to1@p
    pTransformed = pTransformed / pTransformed[2,:]
    return pTransformed

def evaluateHmatrix(p1,p2,H2to1):
    p1_ = d2ToHomoginic(p1)
    p2_ = d2ToHomoginic(p2)
    p2temp = getTransformation(p1_,H2to1)
    error = np.linalg.norm(p2_ - p2temp,ord = 2)
    print('Estimation error: ' + str(error)[:5])
    p2temp = p2temp[:-1,:]
    return p2temp

def PlotEstimatedTransformation(im1,im2,p1,p2,p2reconstructed):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.scatter(p1[0,:],p1[1,:],label='Actual location')
    plt.legend()
    plt.title('incline_L')
    plt.subplot(1,2,2)
    plt.imshow(im2)
    plt.scatter(p2[0,:],p2[1,:],label='Actual location')
    plt.scatter(p2reconstructed[0,:],p2reconstructed[1,:],label='Estimated transformed location')
    plt.legend()
    plt.title('incline_R')

# HW functions:
def getPoints(im1,im2,N):
    print('Select ' + str(N) + ' points')
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.title('incline_L')
    p1 = np.transpose(np.array(plt.ginput(n=N,timeout=0)))
    plt.scatter(np.round(p1[0,:]),np.round(p1[1,:]),c = 'r')
    for i in range(p1.shape[1]):
        plt.annotate(str(i+1), (p1[0,i], p1[1,i]))
    plt.subplot(1,2,2)
    plt.imshow(im2)
    plt.title('incline_R')
    p2 = np.transpose(np.array(plt.ginput(n=N,timeout=0)))
    plt.scatter(np.round(p2[0,:]),np.round(p2[1,:]),c = 'r')
    for i in range(p2.shape[1]):
        plt.annotate(str(i+1), (p2[0,i], p2[1,i]))
    return p1,p2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    A = np.zeros((2*p1.shape[1],9))
    A[np.arange(0,2*p1.shape[1],2),0] = p1[0,:]
    A[np.arange(0,2*p1.shape[1],2),1] = p1[1,:]
    A[np.arange(0,2*p1.shape[1],2),2] = 1
    A[np.arange(1,2*p1.shape[1],2),3] = p1[0,:]
    A[np.arange(1,2*p1.shape[1],2),4] = p1[1,:]
    A[np.arange(1,2*p1.shape[1],2),5] = 1
    A[np.arange(0,2*p1.shape[1],2),6] = - p1[0,:] * p2[0,:]
    A[np.arange(1,2*p1.shape[1],2),6] = - p1[0,:] * p2[1,:]
    A[np.arange(0,2*p1.shape[1],2),7] = - p1[1,:] * p2[0,:]
    A[np.arange(1,2*p1.shape[1],2),7] = - p1[1,:] * p2[1,:]
    A[np.arange(0,2*p1.shape[1],2),8] = - p2[0,:]
    A[np.arange(1,2*p1.shape[1],2),8] = - p2[1,:]
    A_squared = np.transpose(A) @ A
    eigen,vectors = np.linalg.eig(A_squared)
    H2to1 = np.reshape(vectors[:,-1],(3,3))
    if H2to1[0,0] < 0:
        H2to1 = -H2to1
    return H2to1

def warpH(im1, H, out_size):
    im1_index = np.ones((3,im1.shape[0]*im1.shape[1]))
    l1 = im1.shape[0]
    l2 = im1.shape[1]
    H_inv = np.linalg.inv(H)
    for i in range(l1):
        for j in range(l2):
            im1_index[0,i*l1+j] = i # x
            im1_index[1,i*l1+j] = j # y
    im1_index = np.int64(im1_index)
    x = np.linspace(0, l1-1, l1)
    y = np.linspace(0, l2-1, l2)
    kind = 'cubic'
    f1 = scipy.interpolate.interp2d(x, y, np.transpose(im1[:,:,0]), kind=kind)
    f2 = scipy.interpolate.interp2d(x, y, np.transpose(im1[:,:,1]), kind=kind)
    f3 = scipy.interpolate.interp2d(x, y, np.transpose(im1[:,:,2]), kind=kind)
    x_new = np.linspace(0, l1-1, l1*4)
    y_new = np.linspace(0, l2-1, l2*4)
    im1_R_inter = np.transpose(f1(x_new,y_new))
    im1_G_inter = np.transpose(f2(x_new,y_new))
    im1_B_inter = np.transpose(f3(x_new,y_new))
    im1_index_reverse_transform = getTransformation(im1_index,H_inv)
    indx = np.int64(im1_index_reverse_transform[0,:])
    indy = np.int64(im1_index_reverse_transform[1,:])
    warp_im1 = np.zeros((out_size[0],out_size[1],3))
    warp_im1[im1_index[0,:],im1_index[1,:],0] = im1_R_inter[indx,indy]
    warp_im1[im1_index[0,:],im1_index[1,:],1] = im1_G_inter[indx,indy]
    warp_im1[im1_index[0,:],im1_index[1,:],2] = im1_B_inter[indx,indy]
    return np.uint8(np.clip(warp_im1,0,255))

def imageStitching(img1, wrap_img2):
    return panoImg

def ransacH(matches, locs1, locs2, nIter, tol):
    return bestH

def getPoints_SIFT(im1,im2):
    return p1,p2

# %% Main

if __name__ == '__main__':
    print('my_homography')
    im1 =  cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)
    plot_clean_image = False
    if plot_clean_image:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(im1)
        plt.title('incline_L')
        plt.subplot(1,2,2)
        plt.imshow(im2)
        plt.title('incline_R')
    N = 6
    p1,p2 = getPoints(im1,im2,N=N)
    H2to1 = computeH(p1, p2)
    p2reconstructed = evaluateHmatrix(p1,p2,H2to1)
    PlotEstimatedTransformation(im1,im2,p1,p2,p2reconstructed)
    out_size = (2000,2000)
    warp_im1 = warpH(im1, H2to1, out_size)
    plt.figure()
    plt.imshow(warp_im1)
