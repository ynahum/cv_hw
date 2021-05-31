import numpy as np
import matplotlib.pyplot as plt
import cv2
from shared_Q1_Q2 import *
from frame_video_convert import *

# %% Q 2.1

# %% Convert .mp4 video to .jpg frames 

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

# %% Reading 2 frames
frame1_file_path = self_output_dir_path + "/" + "0350.jpg"
frame2_file_path = self_output_dir_path + "/" + "0550.jpg"
self1 = cv2.cvtColor(cv2.imread(frame1_file_path), cv2.COLOR_BGR2RGB)
self2 = cv2.cvtColor(cv2.imread(frame2_file_path), cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10,15))
self_images = [self1, self2]
for i,file in enumerate(self_images):
  ax = fig.add_subplot(1, len(self_images),i+1)
  ax.imshow(self_images[i])
  ax.set_axis_off()

# %% Q2.2

# %% Prepearing the model
# load model
model=torch.hub.load('pytorch/vision:v0.5.0','deeplabv3_resnet101',pretrained=True)
# put in inference mode
model.eval()
# define device
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)

# %% Deep segmentation
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
  # save segmented frame as JPEG file
  cv2.imwrite(os.path.join(segmented_self_output_dir_path, fname), seg_img)

# %% Reading 2 frames
seg_frame1_file_path = segmented_self_output_dir_path + "/" + "0350.jpg"
seg_frame2_file_path = segmented_self_output_dir_path + "/" + "0550.jpg"
seg_self1 = cv2.cvtColor(cv2.imread(seg_frame1_file_path), cv2.COLOR_BGR2GRAY)
seg_self2 = cv2.cvtColor(cv2.imread(seg_frame2_file_path), cv2.COLOR_BGR2GRAY)


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




