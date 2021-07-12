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
def create_ref(im_path, downscale_percent=100):
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
    p1 = np.transpose(np.array(plt.ginput(n=N, timeout=0)))
    #p1 = np.array([[300.7987013, 867.8982684, 140.62554113, 1166.5995671],
    #            [425.47402597, 425.47402597, 1243.65584416, 1209.02380952]])
    #p1 = np.array([[75.90692641, 216.5995671, 35.86363636, 290.19264069],
    #            [108.15800866, 108.15800866, 308.37445887, 301.88095238]])
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

    mh.plotImage(img, "cropped image before warp")

    print(f"p1={repr(p1)}")
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

if __name__ == '__main__':
    print('my_ar')

    #Q2.1
    ref_image = create_ref('my_data/in_book.jpg', downscale_percent=50)
    cv2.imwrite('my_data/ref_book.jpg', cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
    mh.plotImage(ref_image,"reference image after warp")
