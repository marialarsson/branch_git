import numpy as np
import cv2
from cv2 import aruco


# Create dictionary of markers
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# Create images of markers with specific ids
px_size = 98
px_edge = 17
start_id = 200
font = cv2.FONT_HERSHEY_SIMPLEX
#create white square
blank_img = np.zeros((px_size+2*px_edge,px_size+2*px_edge), np.uint8)
blank_img[:] = 255
cv2.putText(blank_img, str(start_id), (int(0.5*px_size+px_edge-30),int(0.5*px_size+px_edge+10)), font, 1, (0,0,0))
# create the five markers
MI = []
for j in range(5):
    id = start_id+j
    marker_img = aruco.drawMarker(dictionary, id, px_size)
    marker_img = np.pad(marker_img,px_edge-1,'constant',constant_values =255)
    if j==4: marker_img = np.pad(marker_img,1,'constant',constant_values =255)
    else: marker_img = np.pad(marker_img,1,'constant',constant_values =0)
    cv2.putText(marker_img, 'ID: '+str(id), (px_edge,px_size+2*px_edge-4), font, 0.4, (0,0,0))
    MI.append(marker_img)
# compose image with blank squares and marker images
img_1 = np.hstack((blank_img,MI[0],blank_img))
img_2 = np.hstack((np.rot90(MI[3]),MI[4],np.rot90(np.rot90(np.rot90(MI[1])))))
img_3 = np.hstack((blank_img,np.rot90(np.rot90(MI[2])),blank_img))
img = np.vstack((img_1,img_2,img_3))

# Show image in window
cv2.imshow('Marker', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
