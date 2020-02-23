import cv2
import  matplotlib.pyplot as plt
IMAGE1 = "/home/dyang/Desktop/5.jpg"
IMAGE2 = "/home/dyang/Desktop/06.jpg"

def readImage(path):
    im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return im


im1 = readImage(IMAGE1)
im2 = readImage(IMAGE2)
cv2.imshow("1",im1)
cv2.imshow("2",im2)
stereo = cv2.StereoBM_create(numDisparities=16,blockSize=15)
disparity = stereo.compute(im2,im1)
plt.imshow(disparity,'gray')
plt.show()
