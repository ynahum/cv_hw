from my_BRIEF import *
import numpy as np
import cv2


# %% 2.4 descriptors matching



def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()

def MatchAndPlot(im1_path, im2_path, brief_match_ratio=0.4):

    im1 = cv2.imread(im1_path)
    im1_g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) / 255
    locs1, desc1 = briefLite(im1_g)

    im2 = cv2.imread(im2_path)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) / 255
    locs2, desc2 = briefLite(im2_g)

    matches = briefMatch(desc1, desc2, brief_match_ratio)

    plotMatches(im1, im2, matches, locs1, locs2)

def main():
    data_path = "../../data"

    MatchAndPlot(f'{data_path}/chickenbroth_01.jpg', f'{data_path}/chickenbroth_04.jpg')
    # testMatch(f'{data_path}/chickenbroth_01.jpg', f'{data_path}/chickenbroth_03.jpg')

    MatchAndPlot(f'{data_path}/incline_L.png', f'{data_path}/incline_R.png', brief_match_ratio=0.4)

    pf_scan_scaled_path = f'{data_path}/pf_scan_scaled.jpg'
    MatchAndPlot(pf_scan_scaled_path, f'{data_path}/pf_desk.jpg')
    MatchAndPlot(pf_scan_scaled_path, f'{data_path}/pf_floor.jpg')
    MatchAndPlot(pf_scan_scaled_path, f'{data_path}/pf_floor_rot.jpg')
    MatchAndPlot(pf_scan_scaled_path, f'{data_path}/pf_pile.jpg')
    MatchAndPlot(pf_scan_scaled_path, f'{data_path}/pf_stand.jpg')

    # self matching
    MatchAndPlot(f'{data_path}/chickenbroth_01.jpg', f'{data_path}/chickenbroth_01.jpg')


if __name__ == "__main__":
    main()

