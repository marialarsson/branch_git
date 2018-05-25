import numpy as np
import cv2
import os
import sys

# Get name_number to identify which item in the dataset that you are working on
# Write in terminal after file name, i.e. python image_reposition.py 3
if len(sys.argv)>1: name_number = sys.argv[1]
else: name_number = 0
name_number = str(name_number).zfill(4)

# Import Images
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', '1_Repostioned')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
img1 = cv2.imread(filename1,0)
img2 = cv2.imread(filename2,0)

# Remap pixels
minimum = 30
maximum = 150
scale = 255/(maximum-minimum)
for i in range(len(img1)):
    for j in range(len(img1[0])):
        if img1[i][j] > minimum and img1[i][j] < maximum : img1[i][j] = (img1[i][j]-minimum)*scale
        if img2[i][j] > minimum and img2[i][j] < maximum : img2[i][j] = (img2[i][j]-minimum)*scale

# Stack both images horizontally
images = np.hstack((img1, img2))

# Save Images for the dataset in folder 2_Remapped
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', '2_Remapped')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
cv2.imwrite(filename1,img1)
cv2.imwrite(filename2,img2)
print 'Saved'

# Display images
cv2.imshow('images',images)
cv2.waitKey(0)
cv2.destroyAllWindows()
