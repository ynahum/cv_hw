# %% Imports
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.interpolate import interp2d
from skimage import color

# %% Functions

# Our functions

def plotImage(im,title):
    plt.figure()
    plt.axis('off')
    plt.imshow(im)
    plt.title(title)


def plotTwoImages(im1,im2,title1,title2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(im1)
    plt.title(title1)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(im2)
    plt.title(title2)


def d2ToHomoginic(p):
    pHom = np.concatenate((p,np.ones((1,p.shape[1]))))
    return pHom


def getForwardTransformation(p,H2to1,epsilon = 1e-14):
    pTransformed = H2to1@p
    pTransformed = pTransformed / (pTransformed[2,:]+epsilon) # to avoid zeros
    return pTransformed


def getInvTransformation(p,H2to1,epsilon = 1e-14):
    pTransformed = np.linalg.pinv(H2to1)@p
    pTransformed = pTransformed / (pTransformed[2,:]+epsilon)
    return pTransformed


def evaluateHmatrix(p1,p2,H2to1):
    p1_ = d2ToHomoginic(p1)
    p2_ = d2ToHomoginic(p2)
    p2temp = getInvTransformation(p1_,H2to1)
    error = np.linalg.norm(p2_ - p2temp,ord = 2)
    print('Estimation error: ' + str(error)[:5])
    p2temp = p2temp[:-1,:]
    return p2temp


def plotEstimatedTransformation(im1, im2, im1_title, im2_title, p1, p2, p2reconstructed):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.scatter(p1[0,:],p1[1,:],label='Actual location')
    plt.legend()
    plt.title(im1_title)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(im2)
    plt.scatter(p2[0,:],p2[1,:],label='Actual location')
    plt.scatter(p2reconstructed[0,:],p2reconstructed[1,:],label='Estimated transformed location')
    plt.legend()
    plt.title(im2_title)
    plt.axis('off')


def getCornersAfterTransform(im1,H):
    corners = np.array([[0, 0,                im1.shape[1],   im1.shape[1]],
                        [0, im1.shape[0],     0,              im1.shape[0]],
                        [1, 1,                1,              1]])
    transofrmedCorners = getInvTransformation(corners,H)[:-1,:]
    indOrder = np.argsort(transofrmedCorners[0])
    leftCorners = transofrmedCorners[:, indOrder[:2]]; rightCorners = transofrmedCorners[:, indOrder[2:]]
    LT = leftCorners[:, leftCorners[1].argmin()]; LB = leftCorners[:, leftCorners[1].argmax()]    
    RT = rightCorners[:, rightCorners[1].argmin()]; RB = rightCorners[:, rightCorners[1].argmax()]
    return LT, LB, RT, RB


def shiftH(im1, H):
    LT, LB, RT, RB = getCornersAfterTransform(im1, H)
    Left = np.int64(np.min((LT[0], LB[0])))
    Right = np.int64(np.max((RT[0], RB[0])))    
    Top = np.int64(np.min((RT[1], LT[1])))    
    Bottom = np.int64(np.max((RB[1], LB[1])))
    corners = (Left, Right, Top, Bottom)
    out_size = (Bottom - Top, Right - Left)
    tMatrix = np.array([[1, 0, Left],[0, 1, Top],[0, 0, 1]])
    HShifted = H @ tMatrix
    return HShifted, out_size,corners


def resizeImg(img1, wrap_img2, corners):
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


def Q1_1_get_points(im1, im2, N, im1_title, im2_title, plot_images=True, get_from_user=True):
    if plot_images:
        plotTwoImages(im1, im2, im1_title, im2_title)
    return getPoints(im1, im2, N, im1_title, im2_title, get_from_user=get_from_user)


def Q1_2_compute_H(im1, im2, im1_title, im2_title, p1, p2, plot_estimated_transform=True):
    H2to1 = computeH(p1, p2)
    p2reconstructed = evaluateHmatrix(p1, p2, H2to1)
    if plot_estimated_transform:
        plotEstimatedTransformation(im1, im2, im1_title, im2_title, p1, p2, p2reconstructed)
    H2to1Shifted,out_size,corners = shiftH(im1, H2to1)
    H = H2to1Shifted.copy()
    return H,out_size,corners


def Q1_3_warp(im1, H, out_size, plot_warp=True, run_both_interp_methods=False):
    warp_im1_linear = warpH(im1, H, out_size)
    warp_im1_linear_title = 'Linear interpolation'
    if run_both_interp_methods:
        warp_im1_cubic = warpH(im1, H, out_size,kind='cubic')
        if plot_warp:
            plotTwoImages(warp_im1_linear, warp_im1_cubic, warp_im1_linear_title, 'Cubic interpolation')
    else:
        if plot_warp:
            plotImage(warp_im1_linear, warp_im1_linear_title)
    return warp_im1_linear


def Q1_4_stitch(img2, warped_img1, corners, plot_stitch=True):
    im1_full, warp_im2_full = resizeImg(img2, warped_img1, corners)
    panoImg = imageStitching(im1_full, warp_im2_full)
    if plot_stitch:
        plotImage(panoImg,'Image  stitching')
    return im1_full, warp_im2_full

def Q1_5_SIFT_matching(im1, im2, N, im1_title, im2_title, plot_matches=True):
    p1, p2 = getPoints_SIFT(im1,im2,N,plot_matches=plot_matches)
    H, out_size, corners = Q1_2_compute_H(im1, im2, im1_title, im2_title, p1, p2, plot_estimated_transform=True)
    warp_im1_linear = Q1_3_warp(im1, H, out_size, plot_warp=True, run_both_interp_methods=False)
    Q1_4_stitch(im2, warp_im1_linear, corners)



#---------------------------

# HW functions:


def getPoints(im1, im2, N, im1_title=None, im2_title=None, get_from_user=True):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
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
        warp_im1[y_out, x_out, channel] = np.array([f(x_value, y_value)[0] for x_value, y_value in zip(x_in, y_in)])
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


def getPoints_SIFT(im1, im2, N=5, plot_matches=False):

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    distance = lambda m: m.distance
    good = sorted(good, key=distance)

    good_size_to_take = min(len(good),N)
    good_lists = []
    for i in range(good_size_to_take):
        good_lists.append([good[i]])

    p1 = [ kp1[m.queryIdx].pt for m in good[:good_size_to_take] ]
    p1 = np.array([*p1]).T
    p2 = [ kp2[m.trainIdx].pt for m in good[:good_size_to_take] ]
    p2 = np.array([*p2]).T

    # cv.drawMatchesKnn expects list of lists as matches.
    if plot_matches:
        img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, good_lists[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plotImage(img3,'SIFT KNN matches')

    return p1,p2

# %% Main

if __name__ == '__main__':
    print('my_homography')
    im1 =  cv2.cvtColor(cv2.imread('data/incline_L.png'), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread('data/incline_R.png'), cv2.COLOR_BGR2RGB)

    im1_title = 'incline_L'
    im2_title = 'incline_R'
    N = 5

    # Q1.1
    #p1, p2 = Q1_1_get_points(im1, im2, N, im1_title, im2_title, plot_images=True, get_from_user=False)

    # Q1.2
    #H, out_size, corners = Q1_2_compute_H(im1, im2, im1_title, im2_title, p1, p2, plot_estimated_transform=True)

    # Q1.3
    #warp_im1_linear = Q1_3_warp(im1, H, out_size, plot_warp=True, run_both_interp_methods=False)

    # Q1.4
    #im1_full, warp_im2_full = Q1_4_stitch(im2, warp_im1_linear, corners)

    # Q1.5
    Q1_5_SIFT_matching(im1, im2, N, im1_title, im2_title, plot_matches=True)
