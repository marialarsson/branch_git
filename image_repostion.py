import numpy as np
import cv2
import os
import sys
from my_definitions import get_branch_center_point, display_point
from my_definitions import overlay_images, get_overlap
from my_definitions import move_branch, rotate_image
from my_definitions import optimize_position, optimize_rotation

# Get name_number to identify which item in the dataset that you are working on
# Write in terminal after file name, i.e. python image_reposition.py 3
if len(sys.argv)>1: name_number = sys.argv[1]
else: name_number = 0
name_number = str(name_number).zfill(4)

# Import Images
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', '0_Raw')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
img1 = cv2.imread(filename1,0)
img2 = cv2.imread(filename2,0)

# Overwrite low values (almost black) with 0 (black)
img1[img1<10]=0
img2[img2<10]=0

#Get center points of branches
(ctr1_i,ctr1_j) = get_branch_center_point(img1)
(ctr2_i,ctr2_j) = get_branch_center_point(img2)
#img1 = display_point(img1,ctr1_i,ctr1_j)
#img2 = display_point(img2,ctr2_i,ctr2_j)

# Move branch in img2 to align its center point with img1
ctr_diff_i = ctr2_i-ctr1_i
ctr_diff_j = ctr2_j-ctr1_j
img2 = move_branch(img2,ctr_diff_i,ctr_diff_j)

# Iterate moving and rotating branch in img2, evaluate the overlap, and pick the best one (the one with the highest overlap)
img2 = optimize_position(img2,img1,15)
img2 = optimize_rotation(img2,img1,5)
img2 = optimize_position(img2,img1,10)
img2 = optimize_rotation(img2,img1,3)
img2 = optimize_position(img2,img1,5)
#ol = get_overlap(img2,img1)
#print 'Overlap',ol

# Save Images for the dataset in folder 1_Repostioned
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', '1_Repostioned')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
cv2.imwrite(filename1,img1)
cv2.imwrite(filename2,img2)
print 'Saved'

# Overlay Images and display to display how well the branch in img1 and img2 are overlapping
img_overlayed = overlay_images(img1,img2)
cv2.imshow('image',img_overlayed)
cv2.waitKey(0)
cv2.destroyAllWindows()
