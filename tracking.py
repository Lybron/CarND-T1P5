import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.ndimage.measurements import label
from sklearn import svm

from moviepy.editor import *
from IPython.display import HTML

def load_test_images():
  images = []
  for path in glob.glob('test_images/*.jpg'):
    img = mpimg.imread(path)
    images.append(img)
    
    return images

test_images = load_test_images()

def convert_color(image, conv='RGB'):
  """Convenience function for converting between RGB and various color spaces."""
    
  img = np.copy(image)
  
  if conv == 'RGB2YCrCb':
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
  elif conv == 'RGB2HSV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  elif conv == 'RGB2LUV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
  elif conv == 'RGB2HLS':
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  elif conv == 'RGB2YUV':
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  else:
    # Retain RGB image
    return img

color_spaces = ['RGB', 'RGB2YCrCb', 'RGB2HSV', 'RGB2LUV', 'RGB2HLS', 'RGB2YUV']


cars = glob.glob('datasets/vehicles/**/*.png', recursive=True)
notcars = glob.glob('datasets/non-vehicles/**/*.png', recursive=True)

example = mpimg.imread(cars[0])

print('Cars: ' + str(len(cars)))
print('Not Cars: ' + str(len(notcars)))
print('Total images:' + str(len(cars) + len(notcars)))
print('Image shape: ' + str(example.shape))
print('Data type: ' + str(type(example)))

# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

COLOR_CSPACE = 'RGB2LUV'
HOG_CSPACE = 'RGB2HLS'
FINAL_ORIENT = 6
FINAL_PIX_PER_CELL = 8
FINAL_CELL_PER_BLOCK = 2
FINAL_SPATIAL_SIZE = (16, 16)
FINAL_HIST_BINS = 8

ystart = 400
ystop = 656
SM_SCALE = 1.0
MED_SCALE = 2.0
LG_SCALE = 3.0

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
  if vis == True:
    features, hog_image = hog(img, orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              transform_sqrt=False,
                              visualise=vis, feature_vector=feature_vec)
    return features, hog_image
  else:
    features = hog(img, orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False,
                   visualise=vis, feature_vector=feature_vec)
    return features

def find_bin_spatial(img, size=(32, 32)):
  color1 = cv2.resize(img[:,:,0], size).ravel()
  color2 = cv2.resize(img[:,:,1], size).ravel()
  color3 = cv2.resize(img[:,:,2], size).ravel()
  return np.hstack((color1, color2, color3))

def find_color_hist(img, nbins=32):
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:,:,0], bins=nbins)
  channel2_hist = np.histogram(img[:,:,1], bins=nbins)
  channel3_hist = np.histogram(img[:,:,2], bins=nbins)
  
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
  # Return the individual histograms, bin_centers and feature vector
  return hist_features

def get_features(files):
  """Extracts color, spatial and HOG features for a list of images."""
  combo = []
  for file in files:
    img = mpimg.imread(file)
    combo.append(extract_combined_features(img, COLOR_CSPACE, HOG_CSPACE))
  return combo

def extract_combined_features(img, color_cspace, hog_cspace):
  """Extracts color, spatial and HOG features for a single image."""
  # Apply color conversion if other than 'RGB'
  color_feature_image = convert_color(img, color_cspace)
  hog_feature_image = convert_color(img, hog_cspace)
  
  # Apply bin_spatial() to get spatial color features
  spatial_features = find_bin_spatial(color_feature_image, size=FINAL_SPATIAL_SIZE)
  
  # Apply color_hist() also with a color space option now
  hist_features = find_color_hist(color_feature_image, nbins=FINAL_HIST_BINS)
  
  # Append the new feature vector to the features list
  
  hog_feat1 = get_hog_features(hog_feature_image[:,:,0], FINAL_ORIENT, FINAL_PIX_PER_CELL, FINAL_CELL_PER_BLOCK)
  hog_feat2 = get_hog_features(hog_feature_image[:,:,1], FINAL_ORIENT, FINAL_PIX_PER_CELL, FINAL_CELL_PER_BLOCK)
  hog_feat3 = get_hog_features(hog_feature_image[:,:,2], FINAL_ORIENT, FINAL_PIX_PER_CELL, FINAL_CELL_PER_BLOCK)
  hogs_feats = np.hstack((hog_feat1, hog_feat2, hog_feat3))
  
  return np.hstack((spatial_features, hist_features, hogs_feats))

car_feats = get_features(cars)
noncar_feats = get_features(notcars)

# Create an array stack of feature vectors
X = np.concatenate((car_feats, noncar_feats)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

car_ind = np.random.randint(0, len(cars))


# Define the labels vector
y = np.hstack((np.ones(len(car_feats)), np.zeros(len(noncar_feats))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',FINAL_SPATIAL_SIZE,'and', FINAL_HIST_BINS,'histogram bins')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = svm.LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t=time.time()
n_predict = 20
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
  """ Extract features using HOG sub-sampling and make predictions."""
    
  # Preserve full image to draw rectangles over
  draw_img = np.copy(img)
  
  """
  Set image value range of 0 to 1 to match those of images used while training the model.
  Data set images are PNG vs. test images/video frames which are JPEG
  """
  img = img.astype(np.float32)/255
  
  # Set vehicle detection region of interest
  img_tosearch = img[ystart:ystop,:,:]
  
  # Convert image to desired color space
  ctrans_tosearch = convert_color(img_tosearch, HOG_CSPACE)
  color_trans = convert_color(img_tosearch, COLOR_CSPACE)
  
  if scale != 1:
    imshape = ctrans_tosearch.shape
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
  
  # HOG Color channels
  ch1 = ctrans_tosearch[:,:,0]
  ch2 = ctrans_tosearch[:,:,1]
  ch3 = ctrans_tosearch[:,:,2]
  
  # Define blocks and steps as above
  nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
  nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
  nfeat_per_block = orient*cell_per_block**2
  
  # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
  window = 64
  nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
  cells_per_step = 2  # Instead of overlap, define how many cells to step
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step
  
  # Compute individual channel HOG features for the entire image
  hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
  
  # A list to store all the windows where vehicles are detected
  car_windows = []
  
  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb*cells_per_step
      xpos = xb*cells_per_step
        
      # Extract HOG for this patch
      hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
      hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
      hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
      hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
      
      xleft = xpos*pix_per_cell
      ytop = ypos*pix_per_cell
      
      # Extract the image patch
      subimg = cv2.resize(color_trans[ytop:ytop+window, xleft:xleft+window], (64,64))
      
      # Get color features
      spatial_features = find_bin_spatial(subimg, size=spatial_size)
      hist_features = find_color_hist(subimg, nbins=hist_bins)
      
      # Scale features and make a prediction
      test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
      
      test_prediction = svc.predict(test_features)
      
      if test_prediction == 1:
        xbox_left = np.int(xleft*scale)
        ytop_draw = np.int(ytop*scale)
        win_draw = np.int(window*scale)
        cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
        
        car_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
  return draw_img, car_windows



DETECTION_THRESHOLD = 3

def add_heat(heatmap, bbox_list):
  # Iterate through list of bboxes
  for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
  # Return updated heatmap
  return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold=DETECTION_THRESHOLD):
  # Zero out pixels below the threshold
  heatmap[heatmap <= threshold] = 0
  
  # Return thresholded map
  return heatmap

def draw_labeled_bboxes(img, labels):
  # Iterate through all detected cars
  for car_number in range(1, labels[1]+1):
    # Find pixels with each car_number label value
    nonzero = (labels[0] == car_number).nonzero()
    
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
      
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
      
    # Draw the box on the image
    cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
  
  # Return the image
  return img


BUFFER_RATE = 10

def highlight_cars(image, windows, threshold):
  
  hot = np.zeros_like(image[:,:,0]).astype(np.float)
  
  # Add heat to each box in box list
  hot = add_heat(hot, windows)
  
  # Apply threshold to help remove false positives
  hot = apply_threshold(hot, threshold)
  
  # Visualize the heatmap when displaying
  heat_map = np.clip(hot, 0, 255)
  
  # Find final boxes from heatmap using label function
  heat_labels = label(heat_map)
  highlight_image = draw_labeled_bboxes(np.copy(image), heat_labels)
  return highlight_image

def get_windows(img):
  sm_img, sm_windows = find_cars(img, ystart, ystop, SM_SCALE, svc, X_scaler, FINAL_ORIENT, FINAL_PIX_PER_CELL, FINAL_CELL_PER_BLOCK, FINAL_SPATIAL_SIZE, FINAL_HIST_BINS)
  med_img, med_windows = find_cars(img, ystart, ystop, MED_SCALE, svc, X_scaler, FINAL_ORIENT, FINAL_PIX_PER_CELL, FINAL_CELL_PER_BLOCK, FINAL_SPATIAL_SIZE, FINAL_HIST_BINS)
  lg_img, lg_windows = find_cars(img, ystart, ystop, LG_SCALE, svc, X_scaler, FINAL_ORIENT, FINAL_PIX_PER_CELL, FINAL_CELL_PER_BLOCK, FINAL_SPATIAL_SIZE, FINAL_HIST_BINS)
  
  windows = sm_windows + med_windows + lg_windows
    
  return windows

def process_frame(frame, buffer=None):
  
  all_windows = []
  
  # If there is no current buffer, create it
  if buffer is None:
    buffer = []
    buffer.append(frame)
    
  # If buffer is non-empty but less than buffer threshold, add current frame
  elif len(buffer) < BUFFER_RATE and len(buffer) > 0:
    for img in buffer:
      all_windows += get_windows(img)
      buffer.append(frame)
  # If buffer is non-empty but equal to or greater than threshold, remove oldest frame, then add current
  elif len(buffer) >= BUFFER_RATE and len(buffer) > 0:
    del(buffer[0])
    for img in buffer:
      all_windows += get_windows(img)
      buffer.append(frame)
  
  # Extract windows from current frame
  all_windows += get_windows(frame)
  
  return highlight_cars(frame, all_windows, DETECTION_THRESHOLD), buffer


def process_video(input_path, output_path):
  clip  = VideoFileClip(input_path)
  
  print("Processing video...")
  print("Clip fps:", clip.fps, "Approx frames:", clip.fps * 50)
  
  start = time.time()
    
  new_frames = []
  
  buffer = None
  
  count = 0
  
  for frame in clip.iter_frames():
    out_img, prev_frames = process_frame(frame, buffer)
    
    new_frames.append(out_img)
    buffer = prev_frames
    
    count = count + 1
    print("Clip:", count)

  new_clip = ImageSequenceClip(new_frames, fps=clip.fps)
  new_clip.write_videofile(output_path, audio=False)
    
  end = time.time()

  print('Video processing time:', round((end-start)/60, 3), 'minutes')
  HTML("""
    <video width="1280" height="720" controls>
    <source src="{0}">
    </video>
    """.format(output_path))


test_input = "test_video.mp4"
test_output = "output_images/video_test.mp4"

process_video(test_input, test_output)

project_input = "project_video.mp4"
project_output = "output_images/video_project.mp4"

#process_video(project_input, project_output)

