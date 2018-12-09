import numpy as np
import cv2
import time
#import picamera as picamera
import io
import os


"""this is very much a work in progress. I'm just taking my best guess
for cv2 and stream stuff. comments indicate what I want to do, but
I'm just getting started."""

experiment_name = "test"
save_dir = "dir"
frame_rate = 1.0
num_cams = 6
filenames = [os.path.join(save_dir, experiment_name + '_cam_%i_'%i) for i
            in range(num_cams) ]

with picamera.PiCamera() as camera:
    stream = io.BytesIO()
    cam_stream.capture(stream, use_video_port = True)

    #get the shape of the images to pass to the cv writer
    h, w = stream.shape()





    fourcc = cv2.VideoWriter_fourcc(*'mp4v')



    h, w, _ = first.shape
    cv_writers = [cv2.VideoWriter(fname, fourcc, frame_rate )]

    for foo in camera.capture_continuous(stream, format='jpeg'):
