from briefOrbCommon import *
import numpy as np
import cv2


# %% 2.5 brief properties

def rotateAndCountMatchBrief(im_g):

    locs_0, desc_0 = briefLite(im_g)
    _, center_0 = rotateImage(im_g, 0)

    thetas = np.arange(-180, 180, 10)
    corrMatchesList = []
    for theta in thetas:
        im_rot, center_rot = rotateImage(im_g, theta)
        locs_rot, desc_rot = briefLite(im_rot)
        matches = briefMatch(desc_0, desc_rot)
        corrMatch = checkRotLocs(
            locs0=locs_0,
            locsTheta=locs_rot,
            matches=matches,
            theta=theta,
            c0=center_0,
            cTheta=center_rot)
        corrMatchesList.append(corrMatch)

    plotMatches(corrMatchesList, thetas, title_append=' using Brief')

def main():
    data_path = "../../data"

    im = cv2.imread(f'{data_path}/model_chickenbroth.jpg')
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255

    rotateAndCountMatchBrief(im_g)


if __name__ == "__main__":
    main()