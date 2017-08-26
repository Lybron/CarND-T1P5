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

def process_video(input_path, output_path):
  clip  = VideoFileClip(input_path)
  
  print("Processing video...")
  print("Clip fps:", clip.fps, "Approx frames:", clip.fps * 50)
  
  start = time.time()
  
  new_frames = []
  
  buffer = None
  
  count = 0
  
  for frame in clip.iter_frames():
    
    new_frames.append(frame)
    
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

#process_video(test_input, test_output)

project_input = "project_video.mp4"
project_output = "output_images/video_project.mp4"

process_video(project_input, project_output)

