import numpy as np
import cv2

def get_branch_center_point(img):
    x = 0
    y = 0
    count = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            item = img[i][j]
            if item!=0:
                x=x+i
                y=y+j
                count=count+1
    x=int(0.5 + x/count)
    y=int(0.5 + y/count)
    return x,y

def display_point(img,i_,j_):
    print 'Center Points at',i_,j_
    size = 5
    for i in range(-size,size):
        for j in range(-size,size):
            ii = i+i_
            jj = j+j_
            img[ii][jj]=255
    return img

def overlay_images(imgA,imgB):
    for i in range(len(imgA)):
        for j in range(len(imgA[0])):
            a = imgA[i][j]
            b = imgB[i][j]
            c = int( (int(a)+int(b))/2+0.5 )
            imgA[i][j]=c
    return imgA

def move_branch(img,diff_i,diff_j):
    # Move in vertical direction
    if diff_i>0:
        for i in range(diff_i):
            img = np.insert(img,len(img),img[-0], axis=0) #insert first row last
            img = np.delete(img,0,axis=0) #delete first row
    if diff_i<0:
        diff_i = abs(diff_i)
        for i in range(diff_i):
            img = np.insert(img,0,img[-1], axis=0) #insert last row first
            img = np.delete(img,len(img)-1,axis=0) #delete last row
    # Move in horizontal direction
    if diff_j>0:
        for i in range(diff_j):
            img = np.insert(img,len(img[0]),img[:,0], axis=1) #insert first column last
            img = np.delete(img,0,axis=1) #delete first column
    if diff_j<0:
        diff_j = abs(diff_j)
        for i in range(diff_j):
            img = np.insert(img,0,img[:,-1], axis=1) #insert last column first
            img = np.delete(img,len(img[0])-1,axis=1) #delete last column
    return img

def get_overlap(imgA,imgB):
    imgA = np.copy(imgA)
    imgB = np.copy(imgB)
    no_overlap = 0
    overlap = 0
    imgA[imgA>0]=1
    imgB[imgB>0]=1
    imgAB = imgA+imgB
    (unique, counts) = np.unique(imgAB,return_counts=True)
    for num,count in zip(unique,counts):
        if num == 1: no_overlap = count
        if num == 2: overlap = count
    total = overlap + int(0.5*no_overlap+0.5)
    overlap = float(overlap)/float(total)
    return overlap

def optimize_position(img,img_match,px):
    img_out = np.copy(img)
    overlap_start = get_overlap(img,img_match)
    overlap = -9999
    for i in range(-px,px):
        for j in range(-px,px):
            img_test = move_branch(np.copy(img),i,j)
            overlap_test = get_overlap(img_test,img_match)
            if overlap_test>overlap:
                overlap = overlap_test
                img_out = img_test
    print 'Overlap', overlap
    return img_out

def optimize_rotation(img,img_match,deg):
    overlap_start = get_overlap(img,img_match)
    # Move branch in image 1 and 2 to center of image
    img = np.copy(img)
    img_match = np.copy(img_match)
    (x,y) = get_branch_center_point(img)
    move_x = x - int(len(img)/2)
    move_y = y - int(len(img[0])/2)
    img = move_branch(img,move_x,move_y)
    img_match = move_branch(img_match,move_x,move_y)
    # Perform test rotations
    overlap = -9999
    for i in range(-deg,deg):
        img_test = rotate_image(np.copy(img),i)
        overlap_test = get_overlap(img_test,img_match)
        if overlap_test>overlap:
            overlap = overlap_test
            img_out = img_test
    # Move branches back to original position
    img_out = move_branch(img_out,-move_x,-move_y)
    # Finished
    print 'Overlap', overlap
    return img_out

def rotate_image(img, ang):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, ang, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


# Set name_number to identify which item in the dataset that you are working on
name_number = '0001'

# Import Images
path = os.path.join(os.path.expanduser('~'), 'Git', 'branch_git', 'Dataset', '0_Raw')
filename1 = name_number+"_depth_1.jpg
filename2 = name_number+"_depth_2.jpg
img1 = cv2.imread(path+'/'+filename1,0)
img2 = cv2.imread(path+'/'+filename2,0)

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

# Overlay Images and display to display how well the branch in img1 and img2 are overlapping
img_overlayed = overlay_images(img1,img2)
cv2.imshow('image',img_overlayed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Images for the dataset in folder 1_Repostioned
path = os.path.join(os.path.expanduser('~'), 'Git', 'branch_git', 'Dataset', '1_Repostioned')
filename1 = name_number+"_depth_1.jpg
filename2 = name_number+"_depth_2.jpg
cv2.imwrite(path+'/'+filename1,img1)
cv2.imwrite(path+'/'+filename2,img2)
print 'Saved'
