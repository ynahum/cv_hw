# %% Imports

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.interpolate import interp2d
from skimage import color

# %% Functions

# Our functions

def plotCleanImages(im1,im2,title1,title2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.title(title1)
    plt.subplot(1,2,2)
    plt.imshow(im2)
    plt.title(title2)

def d2ToHomoginic(p):
    pHom = np.concatenate((p,np.ones((1,p.shape[1]))))
    return pHom

def getForwardTransformation(p,H2to1,epsilon = 1e-14):
    pTransformed = H2to1@p
    pTransformed = pTransformed / (pTransformed[2,:]+epsilon) # to avoide zeros
    return pTransformed

def getInvTransformation(p,H2to1):
    pTransformed = np.linalg.pinv(H2to1)@p
    pTransformed = pTransformed / pTransformed[2,:]
    return pTransformed

def evaluateHmatrix(p1,p2,H2to1):
    p1_ = d2ToHomoginic(p1)
    p2_ = d2ToHomoginic(p2)
    p2temp = getInvTransformation(p1_,H2to1)
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


def GetCornersAfterTransform(im1,H):
    corners = np.array([[0, 0,                im1.shape[1],   im1.shape[1]],
                        [0, im1.shape[0],     0,              im1.shape[0]],
                        [1, 1,                1,              1]])
    transofrmedCorners = getInvTransformation(corners,H)[:-1,:]
    indOrder = np.argsort(transofrmedCorners[0])
    leftCorners = transofrmedCorners[:, indOrder[:2]]; rightCorners = transofrmedCorners[:, indOrder[2:]]
    LT = leftCorners[:, leftCorners[1].argmin()]; LB = leftCorners[:, leftCorners[1].argmax()]    
    RT = rightCorners[:, rightCorners[1].argmin()]; RB = rightCorners[:, rightCorners[1].argmax()]
    return LT, LB, RT, RB

def ShiftH(im1, H):
    LT, LB, RT, RB = GetCornersAfterTransform(im1, H)
    Left = np.int64(np.min((LT[0], LB[0])))
    Right = np.int64(np.max((RT[0], RB[0])))    
    Top = np.int64(np.min((RT[1], LT[1])))    
    Bottom = np.int64(np.max((RB[1], LB[1])))
    corners = (Left, Right, Top, Bottom)
    out_size = (Bottom - Top, Right - Left)
    tMatrix = np.array([[1, 0, Left],[0, 1, Top],[0, 0, 1]])
    HShifted = H @ tMatrix
    return HShifted, out_size,corners

def ResizeImg(img1, wrap_img2, corners):
    # corners = (Left, Right, Top, Bottom)
    y = max(corners[3], img1.shape[0]) - min(corners[2], 0)
    x = max(corners[1], img1.shape[1]) - min(corners[0], 0)
    warp_im2_full = np.zeros((y,x, 3))
    im2_warp_maskInd = np.where(wrap_img2 > 0)
    warp_im2_full[im2_warp_maskInd] = wrap_img2[im2_warp_maskInd]
    im1_full = np.zeros(warp_im2_full.shape)
    im1_maskInd = np.where(img1 > 0)
    im1_full[im1_maskInd[0] - corners[2], im1_maskInd[1] - corners[0], im1_maskInd[2]] = img1[im1_maskInd]
    return np.uint8(im1_full), np.uint8(warp_im2_full)

# HW functions:
def getPoints(im1,im2,N,im1_title=None, im2_title=None, get_from_user=True):

    plt.figure()
    plt.subplot(1, 2, 1)

    plt.imshow(im1)
    if im1_title != None:
        plt.title(im1_title)
    if get_from_user:
        print('Select ' + str(N) + ' points')
        p1 = np.transpose(np.array(plt.ginput(n=N,timeout=0)))
    else:
        p1 = np.array([[451.74341398, 506.34865591, 609.95860215, 702.36747312, 517.54973118],
            [110.99205578, 124.99339987, 183.79904503, 479.22740524, 437.22337298]])
    print(f"p1 points: {p1}")
    plt.scatter(np.round(p1[0,:]),np.round(p1[1,:]),c = 'r')
    for i in range(p1.shape[1]):
        plt.annotate(str(i+1), (p1[0,i], p1[1,i]))

    plt.subplot(1,2,2)
    plt.imshow(im2)
    if im2_title != None:
        plt.title(im2_title)
    if get_from_user:
        p2 = np.transpose(np.array(plt.ginput(n=N,timeout=0)))
    else:
        p2 = np.array([[119.62903226, 177.8344086 , 291.09892473, 391.77849462,201.4311828 ],
            [152.10957527, 167.84075806, 233.91172581, 520.21925269, 498.19559677]])
    print(f"p2 points: {p2}")
    plt.scatter(np.round(p2[0,:]),np.round(p2[1,:]),c = 'r')
    for i in range(p2.shape[1]):
        plt.annotate(str(i+1), (p2[0,i], p2[1,i]))

    return p1,p2

def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    A_cols = 9
    A_rows = 2*p1.shape[1]
    A = np.zeros((A_rows,A_cols))
    A[np.arange(0,A_rows,2),0] = p2[0,:]
    A[np.arange(0,A_rows,2),1] = p2[1,:]
    A[np.arange(0,A_rows,2),2] = 1
    A[np.arange(1,A_rows,2),3] = p2[0,:]
    A[np.arange(1,A_rows,2),4] = p2[1,:]
    A[np.arange(1,A_rows,2),5] = 1
    A[np.arange(0,A_rows,2),6] = - p2[0,:] * p1[0,:]
    A[np.arange(1,A_rows,2),6] = - p2[0,:] * p1[1,:]
    A[np.arange(0,A_rows,2),7] = - p2[1,:] * p1[0,:]
    A[np.arange(1,A_rows,2),7] = - p2[1,:] * p1[1,:]
    A[np.arange(0,A_rows,2),8] = - p1[0,:]
    A[np.arange(1,A_rows,2),8] = - p1[1,:]
    A_squared = np.transpose(A) @ A
    eigen,vectors = np.linalg.eig(A_squared)
    H2to1 = np.reshape(vectors[:,-1],(3,3))
    return H2to1

def warpH(im1, H, out_size,kind='linear'):
    im1_float = im1/255
    warp_im1 = np.zeros((out_size[0],out_size[1],3))
    x_out, y_out = np.meshgrid(np.arange(out_size[1]), np.arange(out_size[0]))
    x_out = x_out.reshape(-1); y_out = y_out.reshape(-1)
    q = np.uint16(d2ToHomoginic(np.concatenate((x_out.reshape(1, -1), y_out.reshape(1, -1)), axis=0)))
    p = getForwardTransformation(q,H)
    x_in = p[0]; y_in = p[1]
    inside = np.where(~((x_in < 0) | (y_in < 0) | (x_in >= im1_float.shape[1]) | (y_in >= im1_float.shape[0])))[0]
    x_in = x_in[inside]; y_in = y_in[inside]
    x_out = x_out[inside]; y_out = y_out[inside]
    for channel in range(im1_float.shape[-1]):
        f = interp2d(np.arange(im1_float.shape[1]), np.arange(im1_float.shape[0]), im1_float[:, :, channel], kind=kind,fill_value=0)
        intepolationArray = np.array([f(XX, YY)[0] for XX, YY in zip(x_in, y_in)])
        warp_im1[y_out, x_out, channel] = intepolationArray
    warp_im1 = (warp_im1 * 255).astype('uint8')
    return warp_im1

def imageStitching(img1, wrap_img2):
    panoImg = np.zeros(img1.shape)
    offset = 5 # avoid noise
    im1_mask = np.sum(img1,axis=2) > offset
    im2_wrap_mask = np.sum(wrap_img2,axis=2) > offset
    panoImg[im1_mask] = img1[im1_mask]
    panoImg[im2_wrap_mask] = wrap_img2[im2_wrap_mask]
    return np.uint8(panoImg)

def ransacH(matches, locs1, locs2, nIter, tol):
    return bestH

def getPoints_SIFT(im1,im2):
    return p1,p2

# %% Main

if __name__ == '__main__':
    print('my_homography')
    im1 =  cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)

    # Q1.1
    plot_clean_image = False
    if plot_clean_image:
        plotCleanImages(im1,im2,title1='incline_L',title2='incline_R')
    N = 5
    p1,p2 = getPoints(im1,im2,N=N,im1_title='incline_L',im2_title='incline_R',get_from_user=False)

    # Q1.2
    H2to1 = computeH(p1, p2)
    p2reconstructed = evaluateHmatrix(p1,p2,H2to1)
    PlotEstimatedTransformation(im1,im2,p1,p2,p2reconstructed)
    H2to1Shifted,out_size,corners = ShiftH(im1, H2to1)
    H = H2to1Shifted.copy()

    # Q1.3
    warp_im1_linear = warpH(im1, H2to1Shifted, out_size)
    plot_both_interpolations = False
    if plot_both_interpolations:
        warp_im1_cubic = warpH(im1, H2to1Shifted, out_size,kind='cubic')
        plotCleanImages(warp_im1_linear,warp_im1_cubic,title1='Linear interpolation',title2='Cubic interpolation')

    # Q1.4
    im1_full, warp_im2_full = ResizeImg(im2, warp_im1_linear,corners)
    panoImg = imageStitching(im1_full, warp_im2_full)
    plt.figure()
    plt.imshow(panoImg)
    plt.title('Image  stitching')
    
    