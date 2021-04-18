from briefOrbCommon import *
import numpy as np
import cv2


# %% 2.6 ORB

def rotateAndCountMatchORB(im_g):

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    im_g_uint8 = (im_g * 255).astype('uint8')
    bgr_img = cv2.cvtColor(im_g_uint8, cv2.COLOR_GRAY2BGR)
    kp_0, desc_0 = orb.detectAndCompute(bgr_img, None)
    _, center_0 = rotateImage(im_g, 0)

    thetas = np.arange(-180, 180, 10)
    corrMatchesList = []
    for theta in thetas:
        im_rot_g, center_rot = rotateImage(im_g, theta)
        im_rot_g_uint8 = (im_rot_g * 255).astype('uint8')
        bgr_rot_img = cv2.cvtColor(im_rot_g_uint8, cv2.COLOR_GRAY2BGR)
        kp_rot, desc_rot = orb.detectAndCompute(bgr_rot_img, None)
        bf_matches = bf.match(desc_0, desc_rot)
        locs_0, locs_rot, matches = openCV2numpy(kp_0, kp_rot, bf_matches)
        corrMatch = checkRotLocs(
            locs0=locs_0,
            locsTheta=locs_rot,
            matches=matches,
            theta=theta,
            c0=center_0,
            cTheta=center_rot)
        corrMatchesList.append(corrMatch)

    plotMatches(corrMatchesList, thetas, title_append=' using ORB')

def main():
    data_path = "../../data"

    im = cv2.imread(f'{data_path}/model_chickenbroth.jpg')
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255

    rotateAndCountMatchORB(im_g)


if __name__ == "__main__":
    main()