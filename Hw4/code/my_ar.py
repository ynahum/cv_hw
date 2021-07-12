import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh
#Add imports if needed:
"""
   Your code here
"""
#end imports

#Add functions here:
"""
   Your code here
"""
#Functions end

# HW functions:
def create_ref(im_path, downscale_percent=100, read_hard_coded_p1=False):
    """
       Your code here
    """
    N=4
    img_pre_scale = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    width = int(img_pre_scale.shape[1] * downscale_percent / 100)
    height = int(img_pre_scale.shape[0] * downscale_percent / 100)
    dim = (width, height)
    if downscale_percent != 100:
        print(f"downscale to {dim}")
    img = cv2.resize(img_pre_scale, dim)
    mh.plotImage(img, "select book corners in the following order:top left, top right, bottom left, bottom right")

    print('Select ' + str(N) + ' points')
    if read_hard_coded_p1:
        print("selection is hard coded and taken from some previous run. it might not be relevant if source image or scale are changed!")
        p1 = np.array([[64.32683983, 339.21861472, 8.04978355, 434.45670996],
                    [156.20995671, 156.20995671, 515.51731602, 511.18831169]])
    else:
        p1 = np.transpose(np.array(plt.ginput(n=N, timeout=0)))

    print(f"p1={repr(p1)}")

    plt.scatter(np.round(p1[0,:]),np.round(p1[1,:]),c = 'b')

    #crop bounding rectangle for less interpolation operations
    x_min = int(np.min(p1, axis=1)[0])
    y_min = int(np.min(p1, axis=1)[1])
    x_max = int(np.max(p1, axis=1)[0]+1) + 1
    y_max = int(np.max(p1, axis=1)[1]+1) + 1
    img = img[y_min:y_max,x_min:x_max]

    p1[0] = p1[0] - x_min
    p1[1] = p1[1] - y_min

    width = x_max - x_min
    height = y_max - y_min

    cv2.imwrite('my_data/in_book_cropped.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    mh.plotImage(img, "cropped image before warp")

    p2 = np.array([[0,width-1,0,width-1],
          [0,0,height-1,height-1]])

    H = mh.computeH(p1, p2)
    H_shifted, out_size, warpped_corners = mh.shiftH(img, H)
    print(f"Start warp to out size: {out_size}")
    print(f"warpped_corners: {warpped_corners}")
    warpped_image = mh.warpH(img, H_shifted, out_size)
    left_offset = p2[0][0] -  warpped_corners[0]
    top_offset = p2[1][0] - warpped_corners[2]
    ref_image = warpped_image[top_offset:(top_offset+height),left_offset:(left_offset+width)]
    return ref_image

def im2im(ref_img, bg_img, fg_img):
    p_ref, p_back = mh.getPoints_SIFT(ref_img, bg_img, N=10, plot_matches=True)
    return None

    # use ransac for match improvement:
    s = 4  # because there are 4 couples to calc H
    e = 0.60  # probability of outliers
    p = 0.9999  # to be "on the safe side"
    nIter = int(np.log(1 - p) / np.log(1 - (1 - e) ** s))
    H_ref2background = mh.ransacH(p_ref, p_back, nIter=nIter, tol=2)

    front_im = cv2.resize(fg_img, ref_img.shape[:2][::-1])
    H, outSize, warpEdges = mh.shiftH(front_im, H_ref2background)
    front_warp = cv2.warpPerspective(front_im, np.linalg.pinv(H), (outSize[1], outSize[0]))

    backgroundEdges = mh.edges(0, bg_img.shape[1], 0, bg_img.shape[0])
    front_warp_big, background, panoramaEdges = \
        mh.prepareToMerge(warpEdges, front_warp, backgroundEdges, bg_img)

    # stitching:
    merged_im = mh.imageStitching(background, front_warp_big)
    plt.figure(), plt.imshow(merged_im), plt.show()
    return merged_im


if __name__ == '__main__':
    print('my_ar')

    ref_path = 'my_data/ref_book.jpg'
    run_all = False

    #Q2.1
    run_Q2_1 = False
    if run_all or run_Q2_1:
        print("Q2.1")
        ref_image = create_ref('my_data/in_book.jpg', downscale_percent=100, read_hard_coded_p1=False)
        cv2.imwrite(ref_path, cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
        mh.plotImage(ref_image,"reference image after warp")

    #Q2.2
    run_Q2_2 = True
    if run_all or run_Q2_2:
        print("Q2.2")
        ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        bg_img = cv2.cvtColor(cv2.imread('my_data/bg.jpg'), cv2.COLOR_BGR2RGB)
        for i in range(3):
            fg_img = cv2.cvtColor(cv2.imread('my_data/fg' + str(i+1) + '.jpg'), cv2.COLOR_BGR2RGB)
            out_img = im2im(ref_img, bg_img, fg_img)
            cv2.imwrite('my_data/im2im' + str(i+1) + '.jpg', cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            mh.plotImage(out_img,f"im2im {i+1}")
