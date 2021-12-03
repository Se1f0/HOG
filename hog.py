import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('sifo.jpg',0)
faces = face_cascade.detectMultiScale(img, 1.4, 4)
for (x, y, w, h) in faces:
    faces = img[y:y + h, x:x + w]
    cv2.imwrite('face1.jpg', faces)
face1 = cv2.imread('face1.jpg',0)
face1 = cv2.resize(face1,(128,128),interpolation = cv2.INTER_AREA)
cv2.imwrite('face1.jpg',face1)
h,w = face1.shape[:2]
print(face1)
print(h,w)

imageG = cv2.copyMakeBorder(face1,1,1,1,1,cv2.BORDER_REFLECT)
plt.imshow(imageG,cmap='gray')
height,width = imageG.shape[:2]
print(height,width)
cv2.imshow('test',imageG)
cv2.waitKey(0)

#gradient x
def gradientX(image):
    gx = np.zeros([h,w])
    for p in range(1,w+1) :
        for l in range(1,h+1) :
            gx[l-1,p-1] = int(image[l+1,p]) - int(image[l-1,p])
    return gx
gx = gradientX(imageG)
print(gx)

#gradient y
def gradientY(image):
    gy = np.zeros([h,w])
    for l in range(1,h+1) :
        for p in range(1,w+1) :
            gy[l-1,p-1] = int(imageG[l,p+1]) - int(imageG[l,p-1])
    return gy
gy = gradientY(imageG)
print(gy)

#Magnitude
def magnitude(gx,gy):
    m = np.zeros([h,w])
    for l in range(h) :
        for p in range(w) :
            m[l,p]= math.sqrt( gx[l,p]**2 + gy[l,p]**2 )
    return m
m = magnitude(gx,gy)
print(m)

#direction
def direction(gx,gy):
    d = np.zeros([h,w])
    for l in range(h) :
        for p in range(w) :
            # d[l,p]= math.floor(math.degrees(math.atan( gy[l,p] / gx[l,p])))
            d[l,p]= math.floor(math.degrees(np.arctan2( gy[l,p],gx[l,p])))
            if d[l,p] < 0 :
                d[l,p] %= 360
    return d
d = direction(gx,gy)
print(d)

bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
np.histogram(d, bins = bins)
hist, bins = np.histogram(d, bins = bins)
print (hist) 
print (bins) 
plt.hist(d.ravel(),bins=bins, color='peru', ec = 'black')
plt.show()

#regions
def create_regions(test_image,bloc_size_r,bloc_size_c):
    regions = []
    for r in range(0,test_image.shape[0], bloc_size_r):
        for c in range(0,test_image.shape[1], bloc_size_c):
            window = test_image[r:r+bloc_size_r,c:c+bloc_size_c]
            regions.append(window)
    return np.array(regions)
bloc_size_r = 8
bloc_size_c = 8
regions = create_regions(d,bloc_size_c,bloc_size_r)
print(regions)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        region_x = math.floor(x/8)
        region_y = math.floor(y/8)
        print(region_x,region_y)
        region = regions[(region_y*16) + region_x]
        print(region)
        bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        np.histogram(region, bins = bins)
        hist, bins = np.histogram(region, bins = bins)
        print (hist) 
        print (bins) 
        plt.hist(region.ravel(),bins=bins, color='peru', ec = 'black')
        plt.show()

cv2.imshow('Image',face1)
cv2.setMouseCallback('Image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()