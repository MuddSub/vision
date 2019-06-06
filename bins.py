import cv2
import numpy as np
import copy
import sys
from matplotlib import pyplot as plt
import glob
import os
import subprocess
import numpy as np
import copy
from scipy import ndimage
from scipy import signal
import matplotlib.patches as patches
import time
import copy
import pylab as pl
import math
import tkinter as tk
#from IPython import display
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from scipy.signal import convolve2d, gaussian, argrelextrema
from PIL import ImageTk, Image
import cv2

from PIL import Image
from scipy.ndimage.interpolation import zoom


###################################################
colorBalanceRatio = 5
lb = []
lc = []
le = []
ld = True
lf = []
lg = []
################################
# helpers and driver
################################


def show(img, msg="image", ana=True):
    cv2.imshow(msg, img)
    if ana:
        analysis(img)
    cv2.waitKey(0)


def show2(img, msg="image2", ana=True):

    cv2.imshow(msg, img/255)
    if ana:
        analysis(img)
    cv2.waitKey(100)


def open(name, path1):
    #"/Users/rongk/Downloads/test.jpg"):
    if name == "d":
        path0 = "/home/dhyang/Desktop/Vision/Vision/bins/"
    #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/images/training15.png"
    #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/03.jpg"
    else:
        path0 = "/Users/rongk/Downloads/visionCode/Vision/test2/"
    path2 = ".jpg"
    path = path0+str(path1)+path2
    img = cv2.imread(path)
    return img


def analysis(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    for i, col in enumerate(("b", "g", "r")):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
######################################
# main program removebackscatter
#######################################


def filter(image, blkSize=10*10, patchSize=8, lamb=10, gamma=1.7, r=10, eps=1e-6, level=5):
    image = np.array(image, np.float32)

    bgr = cv2.split(image)
    #show(bgr[2]/255,"initial red",False)

    # image decomposition, probably key
    decomposed = IDilluRefDecompose(image)
    AL, RL = decomposed[0], decomposed[1]

    RL = FsimpleColorBalance(RL, colorBalanceRatio)  # checked
    # show2(RL,"color corrected reflective") #checked
    bgr = cv2.split(RL)
    #show(bgr[0]/255,"RL blue",False)
    #show(bgr[1]/255,"RL green",False)
    #show(bgr[2]/255,"RL red",False)
    return RL
####################################################
# Img Decompose: weighted image decompose
####################################################


def IDilluRefDecompose(img):
    AList = []
    # illumination
    RList = []
    # reflectance
    bgr = cv2.split(img)
    for cnl in bgr:
        alCnl = copy.deepcopy(cnl)
        rlcnl = copy.deepcopy(cnl)
        maxVal = np.asmatrix(cnl).max()
        k = np.multiply(cnl, .5/maxVal)
        rlcnl = np.multiply(k, rlcnl)
        alCnl = np.subtract(alCnl, rlcnl)
        AList.append(alCnl)
        RList.append(rlcnl)
    Al = cv2.merge(AList)
    Rl = cv2.merge(RList)
    return [Al, Rl]
######################################
# Filter
######################################


def FsimpleColorBalance(img, percent):
    if percent <= 0:
        percent = 5
    img = np.array(img, np.float32)
    rows = img.shape[0]
    cols = img.shape[1]
    chnls = img.shape[2]
    halfPercent = percent/200
    if chnls == 3:
        channels = cv2.split(img)
    else:
        channels = copy.deepcopy(img)
        # Not sure
    channels = np.array(channels)
    for i in range(chnls):
        # find the low and high precentile values based on input percentile
        flat = list(channels[i].flat)
        flat.sort()
        lowVal = flat[int(np.floor(len(flat)*halfPercent))]

        topVal = flat[int(np.ceil(len(flat)*(1-halfPercent)))]
        channels[i] = np.where(channels[i] > lowVal, channels[i], lowVal)
        channels[i] = np.where(channels[i] < topVal, channels[i], topVal)
        channels[i] = cv2.normalize(
            channels[i], channels[i], 0.0, 255.0/2, cv2.NORM_MINMAX)
        channels[i] = np.float32(channels[i])
    result = cv2.merge(channels)
    return result
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def binarization(gray):
    ret, thresh1 = cv2.threshold(gray,150, 255, cv2.THRESH_BINARY_INV)
    thresh1 = cv2.bitwise_not(thresh1)
    return thresh1


def getLines(newImg):
    csums = np.sum(newImg, axis=0)
    csums1 = copy.deepcopy(csums)
    lineLocs = []
    leeway = 100
    for i in range(2):
        lineLocs.append([np.argmin(csums), csums[np.argmin(csums)]])
        lhs = lineLocs[i][0]-leeway
        rhs = lineLocs[i][0]+leeway
        if lhs < 0:
            lhs = 0
        if rhs >= newImg.shape[1]:
            rhs = newImg.shape[1]-1
        csums[lhs:rhs] = 1000000
    if True:
        plt.plot(csums1)
        for i in range(len(lineLocs)):
            plt.axvline(x=lineLocs[i][0], color='r', linewidth=1)
        plt.show()
    newImg = cv2.cvtColor(newImg, cv2.COLOR_GRAY2BGR)
    #error = lineLocs[2][1]-(lineLocs[0][1]+lineLocs[1][1])/2
    error = 0
    return lineLocs, error


def plotLines(lineLocs, original):
    for i in range(2):
        cv2.line(original, (lineLocs[i][0], 0),
                 (lineLocs[i][0], original.shape[0]), (0, 255, 0), 3)
    norm = 0
    center = 0
    for k in range(len(lineLocs)):
        center = center + (50000-lineLocs[k][1])*lineLocs[k][0]
        norm = norm + (50000-lineLocs[k][1])
    #center = (int) (center/norm)
    center = (int)((lineLocs[0][0]+lineLocs[1][0])/2)
    cv2.line(original, (center, 0),
             (center, original.shape[0]), (0, 0, 255), 1)
    return original


def segment(image):
    mdpt = (int)(image.shape[0]/2)
    striph = 150
    return image[mdpt - striph: mdpt + striph, :]


def adjust(image):
    alphah = 0
    alphas = 0
    alphav = 5

    h, s, v = cv2.split(image)
    new_image = np.zeros(image.shape, image.dtype)
    h1, s1, v1 = cv2.split(new_image)

    maximum = h.mean()
    #maximum = h.min()
    print(maximum)
    beta = 127-alphah*maximum  # Simple brightness control
    print(beta)
    h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

    maximum = s.mean()
    print(maximum)
    beta = 127-alphas*maximum  # Simple brightness control
    print(beta)
    s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

    maximum = v.mean()
    beta = 127-alphav*maximum  # Simple brightness control
    print(beta)
    v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

    new_image = cv2.merge([h1, s1, v1])
    return new_image


def boundingRectangle(original,thresh):
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h>1000 and w*h<100000:
            cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)

############################################



def mainImg(img):
    original = img
    origin = copy.deepcopy(original)

    o1 = original

    cv2.imshow("original", origin)
    original = filter(original)
    show2(original, "filtered", False)

    segmented = adjust(original)

    #color filter red
    r = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    redSpace = r[:,:,2]

    #binarization
    redSpace = cv2.bitwise_not(redSpace)
    newImg1 = binarization(redSpace)

    boundingRectangle(o1,newImg1)

    #segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    #r, g, b = cv2.split(segmented)
    cv2.imshow("alpha", segmented)
    cv2.imshow("binarization", newImg1)
    cv2.imshow("background subtraction", redSpace)
    cv2.imshow("result", o1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segmented
####################################################
#########################################################
#####################################################


def main():
    img = open(sys.argv[1], int(sys.argv[2]))
    mainImg(img)
    print("Vision Code")


if __name__ == "__main__":
    main()
