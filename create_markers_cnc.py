import numpy as np
import cv2
from cv2 import aruco

# Create dictionary of markers
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50) #50, 100, 250


# Create images of markers with specific ids
px_size = 98
px_edge = 15

font = cv2.FONT_HERSHEY_SIMPLEX

n = 13
m = 9
count = 0

# create blank marker
blank_img = np.zeros((px_size+2*px_edge,px_size+2*px_edge), np.uint8)
blank_img[:] = 255

rows = []
for i in range(n):
    items = []
    for j in range(m):
        if i==0 or j==0 or i==n-1 or j==m-1:
            id = count
            count = count+1
            marker_img = aruco.drawMarker(dictionary, id, px_size)
            marker_img = np.pad(marker_img,px_edge,'constant',constant_values =255)
            cv2.putText(marker_img, 'ID: '+str(id), (px_edge,px_size+2*px_edge-4), font, 0.4, (0,0,0))
            items.append(marker_img)
        else: items.append(blank_img)
    if len(items)>0: rows.append(np.hstack(items))
board_img = np.vstack(rows)

cv2.imwrite('cnc_markers//cnc_board.png',board_img)

# Show image in window
cv2.imshow('Marker board', board_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
