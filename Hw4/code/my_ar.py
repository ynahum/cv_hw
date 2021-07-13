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
        p1 = np.array([[25.7058435, 320.75822315, 1.64128772, 444.21985718],
                    [77.80397196, 74.66511686, 451.32772917, 413.66146794]])
        #p1 = np.array([[26.75212854, 310.2953728, 1.64128772, 428.52558167],
        #            [77.80397196, 76.75768693, 444.00373393, 411.56889787]])
        #p1 = np.array([[64.32683983, 339.21861472, 8.04978355, 434.45670996],
        #            [156.20995671, 156.20995671, 515.51731602, 511.18831169]])
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

def im2im(ref_img, bg_img, fg_img, N=20,\
          plot_sift_pre_ransac=False, plot_post_ransac=False, plot_and_save_inter_imgs=False):

    if plot_post_ransac:
        p1, p2, kp1, kp2, matches = mh.getPoints_SIFT(\
            ref_img, bg_img, N=N, plot_matches=plot_sift_pre_ransac, return_matches=plot_post_ransac)
    else:
        p1, p2 = mh.getPoints_SIFT(ref_img, bg_img, N=N, plot_matches=plot_sift_pre_ransac)

    # based on RANSAC4Dummies pdf (Tel Aviv university CS) we
    # find the minimum number of iterations by (1-q)^h <= epsilon
    epsilon_probablity_error = 0.001  # alarm rate
    q = 0.05  # our estimate for no outliers from some experiments in SIFT
    nIter = int(np.log(epsilon_probablity_error) / np.log(1 - q))
    # print(f"RANSAC number of iterations {nIter}")
    tol = 2
    H, best_inlier_idxs = mh.ransacH(p1, p2, nIter, tol)
    best_inlier_idxs_reshaped = np.reshape(best_inlier_idxs, -1)
    if plot_post_ransac:
        ransac_matches = np.array(matches)[best_inlier_idxs_reshaped]
        img3 = cv2.drawMatches(ref_img, kp1, bg_img, kp2, ransac_matches, None, flags=2)
        mh.plotImage(img3, 'RANSAC SIFT matches')

    scaled_fg_img = cv2.resize(fg_img, (ref_img.shape[1],ref_img.shape[0]))
    if plot_and_save_inter_imgs:
        mh.plotImage(scaled_fg_img, 'scaled FG image')
        cv2.imwrite(f"my_data/im2im{str(i + 1)}_scaled_fg.jpg", cv2.cvtColor(scaled_fg_img, cv2.COLOR_RGB2BGR))
    shifted_H, out_size, fg_corners = mh.shiftH(scaled_fg_img, H)

    warpped_fg_img = mh.warpH(scaled_fg_img, shifted_H, out_size)
    if plot_and_save_inter_imgs:
        mh.plotImage(warpped_fg_img, 'warpped FG image')
        cv2.imwrite(f"my_data/im2im{str(i + 1)}_warpped_fg.jpg", cv2.cvtColor(warpped_fg_img, cv2.COLOR_RGB2BGR))

    bg_corners = (0, bg_img.shape[1], 0, bg_img.shape[0])

    warpped_fg_full_img, bg_full_img, _ = \
        mh.resizePanorama(bg_img, bg_corners, warpped_fg_img, fg_corners)
    if plot_and_save_inter_imgs:
        mh.plotImage(bg_full_img, 'bg full image')
        cv2.imwrite(f"my_data/im2im{str(i + 1)}_bg_full.jpg", cv2.cvtColor(bg_full_img, cv2.COLOR_RGB2BGR))
        mh.plotImage(warpped_fg_full_img, 'warpped_fg_full')
        cv2.imwrite(f"my_data/im2im{str(i + 1)}_warpped_fg_full.jpg", cv2.cvtColor(warpped_fg_full_img, cv2.COLOR_RGB2BGR))
    return mh.imageStitching(bg_full_img, warpped_fg_full_img)


if __name__ == '__main__':
    print('my_ar')

    ref_path = 'my_data/ref_book.jpg'
    run_all = True

    #Q2.1
    run_Q2_1 = False
    if run_all or run_Q2_1:
        print("Q2.1")
        ref_image = create_ref('my_data/in_book.jpg', downscale_percent=50, read_hard_coded_p1=False)
        cv2.imwrite(ref_path, cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
        mh.plotImage(ref_image,"reference image after warp")

    #Q2.2
    run_Q2_2 = False
    if run_all or run_Q2_2:
        print("Q2.2")
        ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        bg_img = cv2.cvtColor(cv2.imread('my_data/bg.jpg'), cv2.COLOR_BGR2RGB)
        mh.plotImage(bg_img,"background image")
        for i in range(3):
            fg_img = cv2.cvtColor(cv2.imread(f"my_data/fg{str(i+1)}.jpg"), cv2.COLOR_BGR2RGB)
            out_img = im2im(ref_img, bg_img, fg_img, N=25, plot_sift_pre_ransac=False,
                            plot_post_ransac=False, plot_and_save_inter_imgs=False)
            cv2.imwrite(f"../output/im2im{str(i+1)}.jpg", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            mh.plotImage(out_img,f"im2im {i+1}")
