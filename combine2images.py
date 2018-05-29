import numpy as np
import cv2
import os

# Import Images
path_1 = '/home/hiro/git/branch_git/Dataset/Testset/Input/'
path_2 = '/home/hiro/git/branch_git/Dataset/Testset/Label/'

files_1 = os.listdir(path_1)

for filename in files_1:
    # Read images
    img1 = cv2.imread(path_1+filename,0)
    img2 = cv2.imread(path_2+filename,0)
    # Combine images in one array
    images = np.hstack((img1,img2))
    # Save file
    path = '/home/hiro/git/branch_git/pix2pix-tensorflow/datasets/branches/train/'
    cv2.imwrite(path+filename,images)
    print images.shape,'Saved'
