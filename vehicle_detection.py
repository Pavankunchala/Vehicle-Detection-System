import os
import re
import cv2

import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

# get file names of the frames
col_frames = os.listdir('frames/')

# sort file names
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# empty list to store the frames
col_images=[]

for i in col_frames:
    # read the frames
    img = cv2.imread('frames/'+i)
    # append the frames to the list
    col_images.append(img)
    
    
#plot 15th frame
i = 15

for frame in [ i , i+1]:
    plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
    plt.title('frame:'+ str(frame))
    #plt.show()
    

#conver the frames to grayscalle

gray1 = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

#lets find the difference betn both images

plt.imshow(cv2.absdiff(gray2,gray1),cmap='gray')
#plt.show()
    
    
diff_image = cv2.absdiff(gray2,gray1)

#image thresholidng 
ret,thresh = cv2.threshold(diff_image,30,255,cv2.THRESH_BINARY)


#plotting the image thersh

plt.imshow(thresh,cmap='gray')
#plt.show()

#applying image dilation
kernels = np.ones((3,3),np.uint8)
dilated = cv2.dilate(thresh,kernel=kernels,iterations=1)

#plotting the dilated image
plt.imshow(dilated,cmap='gray')
#plt.show()

#creating contours
contours,hirerachy  = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

valid_cntrs = []

for i,cntr in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cntr)
    if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
        valid_cntrs.append(cntr)

# count of discovered contours        
len(valid_cntrs)

dmy = col_images[16].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
plt.imshow(dmy)
plt.show()


