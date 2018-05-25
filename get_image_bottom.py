## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import random

# Configure depth and color streams...
# ...from Camera 1
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('746112061960')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # set depth
        #color_image[depth_image>650]=255
        depth_image[depth_image>650]=0
        #print np.max(depth_image), np.average(depth_image), np.min(depth_image)
        depth_image = depth_image*170
        # flip
        color_image = cv2.flip(color_image, 1 )
        depth_image = cv2.flip(depth_image, 1 )

        # Stack all images horizontally
        #images = np.hstack((color_image,depth_image))

        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', depth_image)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 's'
        ch = cv2.waitKey(25)
        if ch==115:
            #name_number = str(random.randint(0,9999))
            name_number = '0016'
            print name_number
            cv2.imwrite(name_number+"_image_2.jpg",color_image)
            cv2.imwrite(name_number+"_depth_2.jpg",depth_image/255)
            print "Save"
finally:
    # Stop streaming
    pipeline.stop()
