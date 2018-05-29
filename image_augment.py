import numpy as np
import cv2
import random
import os
import sys
from my_definitions import move_branch, rotate_image, scale_image, square_image
from my_definitions import get_last_file_index, update_gradient


# Get name_number to identify which item in the dataset that you are working on
# Write in terminal after file name, i.e. python image_reposition.py 3
if len(sys.argv)>1: name_number = sys.argv[1]
else: name_number = 0
name_number = str(name_number).zfill(4)

# Import Images
path = os.path.join(os.path.expanduser('~'), 'git', 'branch_git_offline', '2_Remapped')
filename1 = path+'/'+name_number+'_depth_1.jpg'
filename2 = path+'/'+name_number+'_depth_2.jpg'
img1 = cv2.imread(filename1,0)
img2 = cv2.imread(filename2,0)

# Make images square
img1 = square_image(img1)
img2 = square_image(img2)

# Agument data by a) rotation, b) scale, and c) exchanging top and bottom image
Images1 = []
Images2 = []
n = 5             #number of extra image pairs
rotation = 180.0  #range of random rotation
scale = 0.20      #range of random scale
gradiation = 30        #random add and subtract to change color
for i in range(n):
    # Generate random parameters
    angle = random.uniform(-rotation,rotation)
    scale_factor = random.uniform(1-scale,1+scale)
    flip = bool(random.getrandbits(1))
    grad1 = random.randint(-gradiation,gradiation)
    grad2 = random.randint(-gradiation,gradiation)
    print "a:", angle, "s:",scale_factor, "f:", flip, "g1:", grad1, "g2:", grad2
    # Rotate
    img_temp_1 = rotate_image(np.copy(img1),angle)
    img_temp_2 = rotate_image(np.copy(img2),angle)
    # Scale
    img_temp_1 = scale_image(img_temp_1,scale_factor)
    img_temp_2 = scale_image(img_temp_2,scale_factor)
    # Color
    img_temp_1 = update_gradient(img_temp_1,(15,250),grad1)
    img_temp_2 = update_gradient(img_temp_2,(15,250),grad2)
    # Swap top and bottom
    if flip==True: img_temp_1, img_temp_2 = img_temp_2, img_temp_1
    # Resize
    img_temp_1 = cv2.resize(img_temp_1, (256, 256))
    img_temp_2 = cv2.resize(img_temp_2, (256, 256))
    # Store in list
    Images1.append(np.copy(img_temp_1))
    Images2.append(np.copy(img_temp_2))

# Stack both images in a grid
Images_hstacked = []
for img1,img2 in zip(Images1, Images2): Images_hstacked.append(np.hstack((img1,img2)))
images = np.vstack(Images_hstacked)
h = len(images)
w = len(images[0])
h_new = 650
w_new = int(h_new*(float(w)/float(h)))
images = cv2.resize(images, dsize=(w_new,h_new)) #resize to fit screen

# Save Images for the dataset in folder 1_Repostioned
path_1 = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', 'Input')
path_2 = os.path.join(os.path.expanduser('~'), 'git', 'branch_git', 'Dataset', 'Label')
index = get_last_file_index(path_1)
for img1,img2 in zip(Images1,Images2):
    index = index+1
    filename_1 = path_1+'/'+str(index).zfill(4)+'.jpg'
    filename_2 = path_2+'/'+str(index).zfill(4)+'.jpg'
    cv2.imwrite(filename_1,img1)
    cv2.imwrite(filename_2,img2)
print 'Saved'

# Display images
#cv2.imshow('images',images)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
