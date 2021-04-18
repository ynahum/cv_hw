from my_BRIEF import *
import numpy as np
import cv2


# %% 2.5 brief properties

def checkRotLocs(locs0, locsTheta, matches, theta,
                 c0, cTheta, th_d=9.0):
    # INPUTS
    #     locs0     - m1x3 matrix of keypoints (x,y,l) of the unrotated image
    #     locsTheta - m2x3 matrix of keypoints (x,y,l) of the rotated image
    #     matches   - px2 matrix of matches indexing into locs0 and locsTheta
    #     theta     - rotation angle in degress
    #     c0        - center of the unrotated image
    #     cTheta    - center of the rotated image
    #     th_d      - threshold distance of matched keypoints in pixels
    # OUTPUTS
    #     corrMatch - number of correct matches
    # keep only the matched keypoints (x,y)
    locs0 = locs0[matches[:,0],:2]
    locsTheta = locsTheta[matches[:,1],:2]
    # rotate the locations at theta=0 and shift them to the new center
    theta = np.deg2rad(theta)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    locs0_rot = (rot_mat@(locs0-c0).T).T+cTheta
    # count the number of correct matches with a distance threshold Td
    corrMatch = np.sum(np.sqrt(np.sum((locs0_rot-locsTheta)**2, 1)) < th_d)
    return corrMatch


def rotateImage(image, theta):
    # rotates an image and calculates the new center pixel
    # INPUTS
    #      image      - HxW image to be rotated
    #      theta      - rotation angle in degrees [0,360]
    # OUTPUTS
    #      image_rot  - H2xW2 rotated image
    #      center_rot - (2,) array of the new center pixel
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -theta, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image_rot = cv2.warpAffine(image, M, (nW, nH))
    center_rot = (np.array(image_rot.shape[:2][::-1])-1)/2
    return image_rot, center_rot

def rotateAndCountMatch(im_path):

    im = cv2.imread(im_path)
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
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

    fig = plt.figure()
    ax = fig.gca()
    ax.bar(thetas, corrMatchesList, width=8)
    ax.set_xlabel('$\Theta$')
    ax.set_ylabel('Correct matches')
    ax.set_title('Correct matches vs $\Theta$')
    ax.grid()
    plt.show()

def main():
    data_path = "../../data"

    rotateAndCountMatch(f'{data_path}/model_chickenbroth.jpg')


if __name__ == "__main__":
    main()