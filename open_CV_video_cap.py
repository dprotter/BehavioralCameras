import numpy as np
import cv2
import time
#import picamera as picamera
import io
import os



experiment_name = "test"
save_dir = "dir"
frame_rate = 1.0
num_cams = 6
filenames = [os.path.join(save_dir, experiment_name + '_cam_%i_'%i) for i
            in range(num_cams) ]

with picamera.PiCamera() as camera:
    stream = io.BytesIO()




    fourcc = cv2.VideoWriter_fourcc(*'mp4v')



    h, w, _ = first.shape
    cv_writers = [cv2.VideoWriter(fname, fourcc, frame_rate )]

    for foo in camera.capture_continuous(stream, format='jpeg'):
