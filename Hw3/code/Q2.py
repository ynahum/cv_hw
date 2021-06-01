import numpy as np
import matplotlib.pyplot as plt
import cv2
from shared_Q1_Q2 import *
from frame_video_convert import *

# %% Q2.1

# Convert .mp4 video to .jpg frames 

self_input_file_path = "./my_data/self.mp4"
self_output_dir_path = "./output/self_frames"

if os.path.exists(self_output_dir_path):
  files = glob.glob(self_output_dir_path + "/*")
  for f in files:
    os.remove(f)
  os.removedirs(self_output_dir_path)
if not os.path.exists(self_output_dir_path):
  os.makedirs(self_output_dir_path)

video_to_image_seq(self_input_file_path, self_output_dir_path)

# Showing 2 frames
self_frame1_name = "0350.jpg"
self_frame2_name = "0550.jpg"
self_frame1_file_path = self_output_dir_path + "/" + self_frame1_name
self_frame2_file_path = self_output_dir_path + "/" + self_frame2_name
self1 = cv2.cvtColor(cv2.imread(self_frame1_file_path), cv2.COLOR_BGR2RGB)
self2 = cv2.cvtColor(cv2.imread(self_frame2_file_path), cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10,15))
self_images = [self1, self2]
for i,file in enumerate(self_images):
  ax = fig.add_subplot(1, len(self_images),i+1)
  ax.imshow(self_images[i])
  ax.set_axis_off()

# %% Q2.2

# Prepearing the model
# load model
model=torch.hub.load('pytorch/vision:v0.5.0','deeplabv3_resnet101',pretrained=True)
# put in inference mode
model.eval()
# define device
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)

# Deep segmentation
segmented_self_output_dir_path = "./output/segmented_self_frames"

if os.path.exists(segmented_self_output_dir_path):
  files = glob.glob(segmented_self_output_dir_path + "/*")
  for f in files:
    os.remove(f)
  os.removedirs(segmented_self_output_dir_path)
if not os.path.exists(segmented_self_output_dir_path):
  os.makedirs(segmented_self_output_dir_path)

input_files = sorted(glob.glob(os.path.join(self_output_dir_path, '*.jpg')))
for filename in input_files:
  img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
  path, fname = os.path.split(filename)
  seg_img = DeepLabSegmentation(img, model, device, 1)
  cv2.imwrite(os.path.join(segmented_self_output_dir_path, fname), seg_img)

# Showing the 2 segmented frames

seg_self_frame1_file_path = segmented_self_output_dir_path + "/" + self_frame1_name
seg_self_frame2_file_path = segmented_self_output_dir_path + "/" + self_frame2_name
seg_self1 = cv2.cvtColor(cv2.imread(seg_self_frame1_file_path), cv2.COLOR_BGR2GRAY)
seg_self2 = cv2.cvtColor(cv2.imread(seg_self_frame2_file_path), cv2.COLOR_BGR2GRAY)


fig = plt.figure(figsize=(10,15))
seg_self_images = [seg_self1, seg_self2]

for i,file in enumerate(seg_self_images):
  ax = fig.add_subplot(1, len(seg_self_images),i+1)
  seg_values, value_counts = np.unique(seg_self_images[i].reshape(-1),axis=0, return_counts=True)
  value_count_sort_ind = np.argsort(value_counts)
  #print(seg_values[value_count_sort_ind])
  #print(value_counts[value_count_sort_ind])
  curClass = seg_values[value_count_sort_ind][-2]
  mask_self = np.zeros_like(seg_self_images[i])
  mask_self[seg_self_images[i] == curClass] = 1
  self_cropped = cropWithRespectToMask(self_images[i], mask_self)
  ax.imshow(self_cropped)
  ax.set_axis_off()

# %% Q2.3

# Convert .mp4 video to .jpg frames

dancing_man_input_file_path = "./data/dancing_man_model.mp4"
dancing_man_output_dir_path = "./output/dancing_man_frames"

if os.path.exists(dancing_man_output_dir_path):
  files = glob.glob(dancing_man_output_dir_path + "/*")
  for f in files:
    os.remove(f)
  os.removedirs(dancing_man_output_dir_path)
if not os.path.exists(dancing_man_output_dir_path):
  os.makedirs(dancing_man_output_dir_path)

video_to_image_seq(dancing_man_input_file_path, dancing_man_output_dir_path)

# semantic segmentation using color segmentation

seg_dancing_man_dir_path = "./output/segmented_dancing_man_frames"

if os.path.exists(seg_dancing_man_dir_path):
  files = glob.glob(seg_dancing_man_dir_path + "/*")
  for f in files:
    os.remove(f)
  os.removedirs(seg_dancing_man_dir_path)
if not os.path.exists(seg_dancing_man_dir_path):
  os.makedirs(seg_dancing_man_dir_path)

input_files = sorted(glob.glob(os.path.join(dancing_man_output_dir_path, '*.jpg')))

def color_segmentation(hsv_img, bg_hsv_low, bg_hsv_high, erode=False, erode_iterations=1, erode_kernel=None):
  mask = cv2.inRange(hsv_img, bg_hsv_low,bg_hsv_high)
  mask_inv = cv2.bitwise_not(mask)
  if erode:
    if erode_kernel is None:
      erode_kernel = np.ones((5, 5), np.uint8)
    mask_inv = cv2.erode(mask_inv, erode_kernel, iterations=erode_iterations)
  seg_hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask_inv)
  return seg_hsv_img, mask_inv

for filename in input_files:
  hsv_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HSV)
  path, fname = os.path.split(filename)
  bg_hsv_green_low = (40, 130, 0)
  bg_hsv_green_high = (70, 255, 255)
  seg_hsv_img, _ = color_segmentation(hsv_img, bg_hsv_green_low, bg_hsv_green_high) 
  cv2.imwrite(os.path.join(seg_dancing_man_dir_path, fname), cv2.cvtColor(seg_hsv_img, cv2.COLOR_HSV2BGR))

dancing_man_frame1_name = "0100.jpg"
dancing_man_frame2_name = "0200.jpg"
dancing_man_frame1_file_path = dancing_man_output_dir_path + "/" + dancing_man_frame1_name
dancing_man_frame2_file_path = dancing_man_output_dir_path + "/" + dancing_man_frame2_name
dancing_man1 = cv2.cvtColor(cv2.imread(dancing_man_frame1_file_path), cv2.COLOR_BGR2RGB)
dancing_man2 = cv2.cvtColor(cv2.imread(dancing_man_frame2_file_path), cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10,15))
dancing_man_images = [dancing_man1, dancing_man2]
for i,file in enumerate(dancing_man_images):
  ax = fig.add_subplot(1, len(dancing_man_images),i+1)
  ax.imshow(dancing_man_images[i])
  ax.set_axis_off()

seg_dancing_man_frame1_file_path = seg_dancing_man_dir_path + "/" + dancing_man_frame1_name
seg_dancing_man_frame2_file_path = seg_dancing_man_dir_path + "/" + dancing_man_frame2_name
seg_dancing_man1 = cv2.cvtColor(cv2.imread(seg_dancing_man_frame1_file_path), cv2.COLOR_BGR2RGB)
seg_dancing_man2 = cv2.cvtColor(cv2.imread(seg_dancing_man_frame2_file_path), cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10,15))
seg_dancing_man_images = [seg_dancing_man1, seg_dancing_man2]
for i,file in enumerate(seg_dancing_man_images):
  ax = fig.add_subplot(1, len(seg_dancing_man_images),i+1)
  ax.imshow(seg_dancing_man_images[i])
  ax.set_axis_off()

# %% Q2.4

# read background image

pink_floyd_bg_file_path = "./my_data/pink_floyd.jpg"
pink_floyd_bg_img = cv2.imread(pink_floyd_bg_file_path)

def scale_image(img, scale_percent):
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def add_fg_image(bg_img, roi_fg, roi_fg_mask, y_offset, x_offset):
  height, width, channels = roi_fg.shape
  roi = bg_img[y_offset:(y_offset+height), x_offset:(x_offset+width)]
  roi_fg_inv_mask = cv2.bitwise_not(roi_fg_mask)
  roi_bg = cv2.bitwise_or(roi, roi, mask=roi_fg_inv_mask)
  final_roi = cv2.add(roi_bg,roi_fg)
  bg_img[y_offset:(y_offset+height), x_offset:(x_offset+width)] = final_roi

# scale, segment and add hand image to background image

hand_file_path = "./my_data/hand.jpg"
hand_img = cv2.imread(hand_file_path)
#print('Original hand dimensions : ',hand_img.shape)
resized_hand_img = scale_image(hand_img, scale_percent=60)
bg_hsv_low = (0,0,238)
bg_hsv_high = (255,14,255)
hsv_hand_img = cv2.cvtColor(resized_hand_img, cv2.COLOR_BGR2HSV)
seg_hsv_hand_img, seg_hand_mask = color_segmentation(hsv_hand_img, bg_hsv_low, bg_hsv_high)
roi_fg = cv2.cvtColor(seg_hsv_hand_img, cv2.COLOR_HSV2BGR)
hand_x_offset = 600
hand_y_offset = 300
add_fg_image(pink_floyd_bg_img, roi_fg, roi_fg_mask=seg_hand_mask, y_offset=hand_y_offset, x_offset=hand_x_offset)

#fig = plt.figure(figsize=(10,15))
#ax = fig.add_subplot(1,1,1)
#ax.imshow(cv2.cvtColor(pink_floyd_bg_img, cv2.COLOR_BGR2RGB))
#ax.set_axis_off()
#ax = fig.add_subplot(1,1,1)
#ax.imshow(cv2.cvtColor(roi_bg, cv2.COLOR_BGR2RGB))
#ax.set_axis_off()
#ax = fig.add_subplot(1,1,1)
#ax.imshow(cv2.cvtColor(seg_hsv_hand_img, cv2.COLOR_HSV2RGB))
#ax.set_axis_off()

# crop, scale and add dancing man to background

final_video_frames_dir_path = "./output/final_video_frames"

if os.path.exists(final_video_frames_dir_path):
  files = glob.glob(final_video_frames_dir_path + "/*")
  for f in files:
    os.remove(f)
  os.removedirs(final_video_frames_dir_path)
if not os.path.exists(final_video_frames_dir_path):
  os.makedirs(final_video_frames_dir_path)

input_files = sorted(glob.glob(os.path.join(dancing_man_output_dir_path, '*.jpg')))
for filename in input_files:

  curr_bg_img = pink_floyd_bg_img.copy()
  img = cv2.imread(filename)

  # horizontal crop
  _, width, _ = img.shape
  cropped_img = img[:, int(width/3):int(width*2/3), :]

  # scale
  scaled_img = scale_image(cropped_img,30)
  #print(scaled_img.shape)

  # color segment
  hsv_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2HSV)
  bg_hsv_green_low = (40, 130, 0)
  bg_hsv_green_high = (70, 255, 255)
  seg_hsv_img, mask = color_segmentation(hsv_img, bg_hsv_green_low, bg_hsv_green_high) 
  seg_img = cv2.cvtColor(seg_hsv_img, cv2.COLOR_HSV2BGR)

  # add to background
  x_offset = 620
  y_offset = 210
  add_fg_image(curr_bg_img, seg_img, mask, y_offset, x_offset)

  path, fname = os.path.split(filename)
  cv2.imwrite(os.path.join(final_video_frames_dir_path, fname), curr_bg_img)

image_seq_to_video(final_video_frames_dir_path, output_path='./output/video.mp4')

#fig = plt.figure(figsize=(10,15))
#ax = fig.add_subplot(1,1,1)
#ax.imshow(cv2.cvtColor(curr_bg_img, cv2.COLOR_BGR2RGB))
#ax.set_axis_off()



