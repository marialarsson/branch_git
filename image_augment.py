import numpy as np
import cv2
import random
import os
import sys
from my_definitions import move_branch, rotate_image


# Get name_number to identify which item in the dataset that you are working on
# Write in terminal after file name, i.e. python image_reposition.py 3
if len(sys.argv)>1: name_number = sys.argv[1]
else: name_number = 0
name_number = str(name_number).zfill(4)

# Import Images
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', '2_Remapped')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
img1 = cv2.imread(filename1,0)
img2 = cv2.imread(filename2,0)

Images1 = []
Images2 = []
Images1.append(img1)
Images2.append(img2)

# Agument data by adding images where the branch is rotaed and scaled
n = 3           #number of extra image pairs
rotation = 30   #range of random rotation
scale = 0.2     #range of random scale
for i in range(n):
    ang = random.randint(-rotation,rotation)
    img_temp_1 = rotate_image(np.copy(img1),)


# Stack both images horizontally
images = np.hstack((img1, img2))

# Save Images for the dataset in folder 1_Repostioned
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', '3_Augmented')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
cv2.imwrite(filename1,img1)
cv2.imwrite(filename2,img2)
print 'Saved'

# Display images
cv2.imshow('images',images)
cv2.waitKey(0)
cv2.destroyAllWindows()
