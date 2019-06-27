import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from skimage.transform import rotate
from scipy.signal import savgol_filter
from scipy import signal
import os
import sys
import time

###################################################
colorBalanceRatio = 20
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
        path0 = "/home/dhyang/Desktop/Vision/Vision/images/"
        #path0 = "/home/dhyang/Desktop/Vision/Vision/Neural_Net/Test/"
    #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/images/training15.png"
    #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/03.jpg"
    else:
        path0 = "/Users/rongk/Downloads/visionCode/Vision/test2/"
    path = path0+str(path1)
    if os.path.isfile(path+'.jpg'):
        img = cv2.imread(path+'.jpg')
    else:
        img = cv2.imread(path+'.png')
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

def detectNoteheadBlobs(img, minarea, maxarea):

    # define blob detector
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 100;
    # params.maxThreshold = 200;

    # Filter by Area
    # params.filterByArea = True
    params.minArea = minarea
    params.maxArea = maxarea

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(np.array(img), keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, im_with_keypoints

def reflect(image, blkSize=10*10, patchSize=8, lamb=10, gamma=1, r=10, eps=1e-6, level=5):
    start_time = time.time()
    image = np.array(image, np.float32)
    bgr = cv2.split(image)
    RL = IDilluRefDecompose(image)
    #print("1:",time.time()-start_time)
    RL = FsimpleColorBalance(RL, colorBalanceRatio)  # checked
    #print("2:",time.time()-start_time)
    bgr = cv2.split(RL)
    return RL
####################################################
# Img Decompose: weighted image decompose
####################################################


def IDilluRefDecompose(img):
    RList = []
    bgr = cv2.split(img)
    for cnl in bgr:
        rlcnl = copy.deepcopy(cnl)
        maxVal = np.asmatrix(cnl).max()
        k = np.multiply(cnl, .5/maxVal)
        rlcnl = np.multiply(k, rlcnl)
        RList.append(rlcnl)
    Rl = cv2.merge(RList)
    return Rl
######################################
# Filter
######################################


def FsimpleColorBalance(img, percent):
    start_time = time.time()
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
    print("1:",time.time()-start_time)

    for i in range(chnls):
        # find the low and high precentile values based on input percentile
        flat = np.array(channels[i].flat)
        flat.sort()
        lowVal = flat[int(np.floor(len(flat)*halfPercent))]

        topVal = flat[int(np.ceil(len(flat)*(1-halfPercent)))]
        channels[i] = np.where(channels[i] > lowVal, channels[i], lowVal)
        channels[i] = np.where(channels[i] < topVal, channels[i], topVal)
        channels[i] = cv2.normalize(
            channels[i], channels[i], 0.0, 255.0/2, cv2.NORM_MINMAX)
        channels[i] = np.float32(channels[i])
    print("2:",time.time()-start_time)

    result = cv2.merge(channels)
    return result
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def binarization(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray,9,10,15)
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 2)
    #ret, thresh1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    thresh1 = cv2.bitwise_not(thresh1)
    #thresh1 = cv2.bilateralFilter(thresh1,4,5,75)


    return thresh1


def getLines(newImg,graph):
    newImg = 255-newImg
    csums = np.sum(newImg, axis=0)
    csums1 = copy.deepcopy(csums)
    lineLocs = []
    leeway = 25
    f = savgol_filter(csums1,101,2,0)
    #csums = np.subtract(csums,f)
    #csums = np.convolve(csums,[2,-1])
    csums2 = copy.deepcopy(csums)
    #for i in range(len(csums)):
    #    if i > 0:
    #        csums[i] = csums2[i]+csums2[i-1]
    #csums2 = copy.deepcopy(csums)

    for i in range(2):
        pred = np.argmax(csums)
        c1= newImg[:,pred]
        m= (int)(np.sum(c1)/255)
        if m<=30:
            continue

        lineLocs.append([pred, csums[np.argmax(csums)]])
        lhs = lineLocs[i][0]-leeway
        rhs = lineLocs[i][0]+leeway
        if lhs < 0:
            lhs = 0
        if rhs >= newImg.shape[1]:
            rhs = newImg.shape[1]-1
        csums[lhs:rhs] = 0
    if len(lineLocs)==2:
        c1= newImg[:,lineLocs[0][0]]
        c2 = newImg[:,lineLocs[1][0]]

        m1 = np.sum(c1)
        m2 = np.sum(c2)

        w1 = np.arange(len(c1))
        w2 = np.arange(len(c2))

        com1 = (int)(np.sum(np.multiply(c1,w1))/m1)
        com2 = (int)(np.sum(np.multiply(c2,w2))/m2)

        I1 = 0
        I2 = 0

        l1 = 0
        l2 = 0
        for i in range(len(c1)):
            if c1[i]!=0:
                l1+=1
                I1 +=abs(c1[i]-com1)
            if c2[i]!=0:
                l2+=1
                I2 +=abs(c2[i]-com2)
        I1 = I1/l1
        I2 = I2/l2
        print(m1/255,m2/255)
        print(com1,com2)
        print(I1,I2)



    if graph:
        plt.plot(csums2)
        for i in range(len(lineLocs)):
            plt.axvline(x=lineLocs[i][0], color='r', linewidth=1)
        plt.ioff()
        plt.show()
    newImg = cv2.cvtColor(newImg, cv2.COLOR_GRAY2BGR)
    #error = lineLocs[2][1]-(lineLocs[0][1]+lineLocs[1][1])/2
    error = 0
    return lineLocs, error


def plotLines(lineLocs, original):
    for i in range(len(lineLocs)):
        cv2.line(original, (lineLocs[i][0], 0),
                 (lineLocs[i][0], original.shape[0]), (0, 255, 0), 3)
    norm = 0
    center = 0
    for k in range(len(lineLocs)):
        center = center + (50000-lineLocs[k][1])*lineLocs[k][0]
        norm = norm + (50000-lineLocs[k][1])
    #center = (int) (center/norm)
    if len(lineLocs)==2:
        center = (int)((lineLocs[0][0]+lineLocs[1][0])/2)
        cv2.line(original, (center, 0),
             (center, original.shape[0]), (0, 0, 255), 1)
    return original


def segment(image):
    mdpt = (int)(image.shape[0]/2)
    striph = 100
    return image[mdpt - striph: mdpt + striph, :]


def adjust(image):
    alphah = 3
    alphas = 6
    alphav = 3

    h, s, v = cv2.split(image)
    new_image = np.zeros(image.shape, image.dtype)
    h1, s1, v1 = cv2.split(new_image)

    maximum = h.mean()
    #maximum = h.min()
    beta = 127-alphah*maximum  # Simple brightness control
    h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

    maximum = s.mean()
    beta =127-alphas*maximum  # Simple brightness control
    s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

    maximum = v.mean()
    beta = 127-alphav*maximum  # Simple brightness control
    v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

    new_image = cv2.merge([h1, s1, v1])
    return new_image


def adjustLAB(image):
    alphah = 0
    alphas = 0
    alphav = 1

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    h, s, v = cv2.split(image)
    new_image = np.zeros(image.shape, image.dtype)
    h1, s1, v1 = cv2.split(new_image)

    maximum = h.mean()
    #maximum = h.min()
    beta = -alphah*maximum  # Simple brightness control
    h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

    maximum = s.mean()
    beta = 127-alphas*maximum  # Simple brightness control
    s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

    maximum = v.mean()
    beta = -alphav*maximum  # Simple brightness control
    v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)



    new_image = cv2.merge([h1, s1, v1])
    new_image = cv2.cvtColor(new_image,cv2.COLOR_YUV2BGR);
    return new_image

############################################

def HoughLines(gray):
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=0,lines=np.array([]), minLineLength = 30,maxLineGap=4)
    a,b,c = lines.shape
    gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for i in range(a):
        cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("lsdjjndldsjd",gray)


def rotateToHorizontal(img, lb=-20, ub=20, incr=.5, topN=2):
    bestscore = -np.inf
    bestTheta = 0
    for theta in np.arange(lb, ub, incr):
        imgRot = rotate(img,theta)
        csums = np.sum(imgRot, axis=0)
        csums_sorted = sorted(csums)[::-1]
        curscore = np.sum(csums_sorted[0:topN])
        if curscore > bestscore:
            bestscore = curscore
            bestTheta = theta
    result = rotate(img,bestTheta)
    print(bestTheta)
    return result

def findLeft(img):
    ans = []
    ans1 = []
    for i in range(len(img)):
        if i<len(img)-2:
            ans.append(np.subtract(img[i+2],np.add(img[i+1],img[i])))
        if i > 1:
            ans1.append(np.subtract(img[i-2],np.add(img[i],img[i-1])))
    csums1 = np.sum(ans1,axis=0)
    csums = np.sum(ans,axis = 0)
    plt.plot(csums1)
    plt.show()

    leeway = 20
    lineLocs = []
    for i in range(2):
        lineLocs.append([np.argmax(csums), csums[np.argmax(csums)]])
        lhs = lineLocs[i][0]-leeway
        rhs = lineLocs[i][0]+leeway
        if lhs < 0:
            lhs = 0
        if rhs >= img.shape[1]:
            rhs = img.shape[1]-1
        csums[lhs:rhs] = 0
    for i in range(2,4):
        lineLocs.append([np.argmax(csums1), csums[np.argmax(csums1)]])
        lhs = lineLocs[i][0]-leeway
        rhs = lineLocs[i][0]+leeway
        if lhs < 0:
            lhs = 0
        if rhs >= img.shape[1]:
            rhs = img.shape[1]-1
        csums1[lhs:rhs] = 0
    return lineLocs

def mainImg(img):
    start_time = time.time()
    original = img
    origin = copy.deepcopy(original)

    o1 = original

    segmented = segment(original)

    #segmented = reflect(original)


    segmented = adjustLAB(segmented)

    print(time.time()-start_time)

    segmented = adjust(segmented)
    # Higher discernability = lower distinguishing power

    discernability = 25

    newImg = cv2.medianBlur(segmented, discernability)
    newImg = 255-cv2.absdiff(segmented, newImg)

    #newImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
    #newImg1 = binarization(newImg)
    #newImg1 = cv2.fastNlMeansDenoisingColored(newImg,None,10,0,7,21)
    print(time.time()-start_time)

    newImg1 = binarization(newImg)
    newImg1 = 255-newImg1
    #newImg1 = np.multiply(newImg1,mask)
    newImg1 = 255-newImg1

    for i in range(1):
        newImg1 = cv2.dilate(newImg1,np.ones((5,1)),iterations = 2)
        newImg1 = cv2.erode(newImg1,np.ones((5,1)),iterations = 1)
    newImg1 = cv2.erode(newImg1,np.ones((5,1)),iterations = 1)
    #newImg1 = cv2.dilate(newImg1,np.ones((1,3)),iterations=1)
    newImg1 = cv2.erode(newImg1,np.ones((1,3)),iterations = 1)

    print(time.time()-start_time)

    #newImg1 = rotateToHorizontal(newImg1)
    #lineLocs = findLeft(newImg1)
    #newImg1 = cv2.bilateralFilter(newImg1,9,75,75)

    #experimental code: blob subtraction
    #newImg1_inv = cv2.bitwise_not(newImg1)
    #newImg2 = cv2.multiply(newImg1_inv, mask)
    #newImg2 = cv2.bitwise_not(newImg2)


    lineLocs, certainty = getLines(newImg1,True)
    newImg1 = plotLines(lineLocs, newImg1)
    o1 = plotLines(lineLocs, o1)

    #HoughLines(newImg1)
    cv2.imshow("original",origin)
    cv2.imshow("alpha", segmented)

    #plt.imshow(newImg1)
    cv2.imshow("binarization", newImg1)
    #cv2.imshow("mask", mask)
    #cv2.imshow("multiplied",newImg2)
    #cv2.imshow("background subtraction", newImg)
    #cv2.imshow("result", o1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return newImg1
####################################################
#########################################################
#####################################################


def main():
    if sys.argv[2]=='all':
        for i in range(1,126):
            img = open(sys.argv[1], i)
            b = mainImg(img)
            cv2.imwrite("/home/dhyang/Desktop/Vision/Vision/Neural_Net/Train_binarized/"+str(i)+".jpg",b)
    else:
        img = open(sys.argv[1], sys.argv[2])
        b = mainImg(img)
        cv2.imshow("Result",img)
        #cv2.imshow("binarized",b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
