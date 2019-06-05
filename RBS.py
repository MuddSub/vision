# from RemoveBackScatter and Java codes by Isaac Changhua
import cv2
import numpy as np
import copy
import sys
from matplotlib import pyplot as plt
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
def show(img,msg="image",ana=True):
	cv2.imshow(msg,img)
	if ana:
		analysis(img)
	cv2.waitKey(0)
def show2(img,msg="image2",ana = True):

	cv2.imshow(msg,img/255)
	if ana:
		analysis(img)
	cv2.waitKey(100)
def test(name,path1):
	#"/Users/rongk/Downloads/test.jpg"):
	if name == "d":
		path0 ="/home/dhyang/Desktop/Vision/Vision/test1/"
	#path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/images/training15.png"
	#path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/03.jpg"
	else:
		path0 = "/Users/rongk/Downloads/visionCode/Vision/test2/"
	path2=".jpg"
	path = path0+str(path1)+path2
	print(path)
	img = cv2.imread(path)

	cv2.imshow("testing begins initial",img)
	img2 = filter(img)
	#show2(img2,'final product')
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
def drive(name,path1):
	#"/Users/rongk/Downloads/test.jpg"):
	if name == "d":
		path0 ="/home/dhyang/Desktop/Vision/Vision/test2/"
	#path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/images/training15.png"
	#path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/03.jpg"
	else:
		path0 = "/Users/rongk/Downloads/visionCode/Vision/test2/"
	path2=".jpg"
	path = path0+str(path1)+path2
	img = cv2.imread(path)
	cv2.imshow("initial",img)
	img2 = enhance(img)
	#show2(img2,'final product')
	#cv2.waitKey(0)
	cv2.destroyAllWindows()

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
    show(bgr[2]/255, "initial red", False)

    # image decomposition, probably key
    decomposed = IDilluRefDecompose(image)
    AL, RL = decomposed[0], decomposed[1]
    show2(AL, "pre illuminance")
    show2(RL, "pre Reflective")  # checked

    RL = FsimpleColorBalance(RL, colorBalanceRatio)  # checked
    show2(RL, "color corrected reflective")  # checked

    bgr = cv2.split(RL)
    show(bgr[0]/255, "RL blue", False)
    show(bgr[1]/255, "RL green", False)
    show(bgr[2]/255, "RL red", False)

    # Calculate the air-light using Airlight Esitmate
    airlight = AEestimate(AL, blkSize)
    # print(airlight)
    #show2(airlight,"backgroun light sample",False)
    # estiamte the transimission map

    fTrans = 0.6
    trans = TEtransEstimate(AL, patchSize, airlight,
                            lamb, fTrans, r, eps, gamma)
    show2(trans*255, "final transmission map", False)  # work!

    # correct, np array adding matters!
    AL = dehazeProcess(AL, trans, airlight)
    show2(AL, "dehaze illuminance")

    bgr = cv2.split(AL)
    show(bgr[0]/255, "AL blue", False)
    show(bgr[1]/255, "AL green", False)
    show(bgr[2]/255, "AL red", False)

    # maybe just set all squares in transmission map 0?
    AL2 = AL-2*np.array(cv2.merge((trans, trans, trans)))
    show2(AL2, "AL2")
#############################################################################################


def enhance(image, blkSize=10*10, patchSize=8, lamb=10, gamma=1.7, r=10, eps=1e-6, level=5):
    image = np.array(image, np.float32)
    # image decomposition
    decomposed = IDilluRefDecompose(image)
    AL, RL = decomposed[0], decomposed[1]
    show2(AL, "pre illuminance")
    show2(RL, "pre Reflective")  # checked
    RL = FsimpleColorBalance(RL, colorBalanceRatio)  # checked
    show2(RL, "color corrected reflective")  # checked
    # Calculate the air-light using Airlight Esitmate
    airlight = AEestimate(AL, blkSize)
    # print(airlight)
    #show2(airlight,"backgroun light sample",False)
    # estiamte the transimission map
    fTrans = 0.6
    trans = TEtransEstimate(AL, patchSize, airlight,
                            lamb, fTrans, r, eps, gamma)
    show2(trans*255, "final transmission map", False)  # work!
    # correct, np array adding matters!
    AL = dehazeProcess(AL, trans, airlight)
    show2(AL, "dehaze illuminance")
    # calculate weight
    w1 = calWeight(AL)
    w2 = calWeight(RL)
    # Fuse # correct up to here
    cv2.waitKey(0)
    return pyramidFuse(w1, w2, AL, RL, level)


def pyramidFuse(w1, w2, img1, img2, level):
    # Normalized weight, use np matrix
    sumW = w1+w2
    w1 = np.divide(w1, sumW)
    w1 = np.multiply(w1, 2)
    w2 = np.divide(w2, sumW)
    w2 = np.multiply(w2, 2)
    # pyramid decomposition and reconstruct
    return IDfuseTwoImage(w1, img1, w2, img2, level)


# different,maybe give the other alg-->simplestColor another try
def dehazeProcess(img, trans, airlight):
    global lf
    lf = []
    balancedImg = FsimpleColorBalance(img, 5)
    #show2(balancedImg,"balanced Img")
    bCnl, gCnl, rCnl = cv2.split(balancedImg)
    #show2(bCnl,"pre bCnl",False)
    #show2(gCnl,"pre gCnl",False)
    #show2(rCnl,"pre rCnl",False)
    # get mean value correct?
    bMean = np.mean(bCnl)
    gMean = np.mean(gCnl)
    rMean = np.mean(rCnl)
    print(bMean)
    print(gMean)
    print(rMean)
    # get transimission map for each channel
    num = max(bMean, gMean, rMean)
    Tb = np.multiply(copy.deepcopy(trans), num/bMean*.8)
    Tg = np.multiply(copy.deepcopy(trans), num/gMean*.9)
    Tr = np.multiply(copy.deepcopy(trans), num/rMean*.8)
    # show(Tb,"Tb",False)
    # show(Tg,"Tg",False)
    # show(Tr,"Tr",False)
    # print([num/bMean*.8,num/gMean*.9,num/rMean*.8])
    # print(airlight)
    # dehaze by formula
    bCnl, gCnl, rCnl = np.array(bCnl), np.array(gCnl), np.array(rCnl)
    # blue
    # np add does mod, we need saturation so use where
    bChannel = (bCnl-airlight[0])/Tb + airlight[0]  # this is wrong
    extractCond = bChannel > 255
    bChannel = np.where(extractCond, 255, bChannel)
    extractCond = bChannel < 0
    bChannel = np.where(extractCond, 0, bChannel)
    # show2(bChannel,"bChannel",False)
    # green
    gChannel = (gCnl-airlight[1])/Tg + airlight[1]  # somehow this is wrong...
    extractCond = gChannel > 255
    gChannel = np.where(extractCond, 255, gChannel)
    extractCond = gChannel < 0
    gChannel = np.where(extractCond, 0, gChannel)
    # show2(gChannel,"gChannel",False)
    lf = gChannel
    # red
    rChannel = (rCnl-airlight[2])/Tr + airlight[2]  # this is right
    extractCond = rChannel > 255
    rChannel = np.where(extractCond, 255, rChannel)
    extractCond = rChannel < 0
    rChannel = np.where(extractCond, 0, rChannel)
    # show2(rChannel,"rChannel",False)
    #show2(bChannel,"dehaze bChannel",False)
    dehazed = cv2.merge((bChannel, gChannel, rChannel))
    return dehazed


def calWeight(img):
    img = np.uint8(img)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    L = np.float32(L)
    L = np.divide(L, 255)
    # calculate luminance weight
    WC = FWluminanceWeight(img, L)
    WC = np.float32(WC)
    # calculate saliency weight
    WS = FWsaliency(img)
    WS = np.float32(WS)
    # calculate exposedness weight
    WE = FWexposedness(L)
    WE = np.float32(WE)
    # sum
    weight = WC + WS + WE
    # cv2.imshow("weight",weight/255)
    # cv2.waitKey(0)
    # show2(weight,"weight",False)
    return weight  # correct
# end of main code


def IDbuildGaussianPyramid(img, level):
    #img = np.array(img,np.float64)
    mask = IDfilterMask(img)
    tmp = cv2.filter2D(img, -1, mask)
    gaussPyr = [0 for i in range(level)]
    gaussPyr[0] = copy.deepcopy(tmp)
    tmpImg = copy.deepcopy(img)
    for i in range(1, level):
        tmpImg = cv2.resize(src=tmpImg, dsize=None, fx=0.5,
                            fy=0.5, interpolation=cv2.INTER_LINEAR)
        # lack new Size() not sure what to put
        tmp = cv2.filter2D(tmpImg, -1, mask)
        gaussPyr[i] = copy.deepcopy(tmp)
    return gaussPyr


def IDfilterMask(img):
    h = np.array([1/16, 4/16, 6/16, 4/16, 1/16])
    row = [0 for i in range(len(h))]
    mat = [copy.deepcopy(row) for i in range(len(h))]
    for i in range(len(h)):
        for j in range(len(h)):
            mat[i][j] = h[i]*h[j]
    return np.array(mat)


def IDbuildLaplacianPyramid(img, level):
    lapPyr = [copy.deepcopy(img) for i in range(level)]
    tmpImg = copy.deepcopy(img)
    for i in range(1, level):
        tmpImg = cv2.resize(src=tmpImg, dsize=None, fx=0.5,
                            fy=0.5, interpolation=cv2.INTER_LINEAR)
        lapPyr[i] = copy.deepcopy(tmpImg)
        cv2.waitKey(0)
    for i in range(0, level-1):
        x, y = lapPyr[i].shape[:2]
        tmpPyr = cv2.resize(
            src=lapPyr[i+1], dsize=(y, x), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        # lb.append(lapPyr[i])
        # lg.append(tmpPyr)
        lapPyr[i] = lapPyr[i]-tmpPyr
        extractCond = lapPyr[i] > 255
        lapPyr[i] = np.where(extractCond, 255, lapPyr[i])
        extractCond = lapPyr[i] < 0
        lapPyr[i] = np.where(extractCond, 0, lapPyr[i])
        # le.append(lapPyr[i])
        #show2(lapPyr[i],"lap done "+str(i),False)
    # for i in range(5):
    #	s = str(i) + "  lap Pyr"
    #	show(lapPyr[i],s,False)
    return lapPyr


def IDreconstructLaplacianPyramid(pyramid):
    level = len(pyramid)
    for i in [level-1-i for i in range(0, level-1)]:
        x, y = pyramid[i-1].shape[:2]
        tmpPyr = cv2.resize(pyramid[i], (y, x), 0, 0, cv2.INTER_LINEAR)
        pyramid[i-1] = pyramid[i-1]+tmpPyr

        extractCond = pyramid[i-1] > 255
        pyramid[i-1] = np.where(extractCond, 255, pyramid[i-1])
        extractCond = pyramid[i-1] < 0
        pyramid[i-1] = np.where(extractCond, 0, pyramid[i-1])
    return pyramid[0]


def IDfuseTwoImage(w1, img1, w2, img2, level):
    global le, lg, lb
    weight1 = np.array(IDbuildGaussianPyramid(w1, level))
    weight2 = np.array(IDbuildGaussianPyramid(w2, level))

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    bgr = cv2.split(img1)
    # good above. not good below
    bCnl1 = np.array(IDbuildLaplacianPyramid(bgr[0], level))
    gCnl1 = np.array(IDbuildLaplacianPyramid(bgr[1], level))
    rCnl1 = np.array(IDbuildLaplacianPyramid(bgr[2], level))
    # ok but not below

    bgr2 = cv2.split(img2)

    bCnl2 = np.array(IDbuildLaplacianPyramid(bgr2[0], level)	)
    gCnl2 = np.array(IDbuildLaplacianPyramid(bgr2[1], level)	)
    rCnl2 = np.array(IDbuildLaplacianPyramid(bgr2[2], level))

    bCnl = (bCnl1*weight1+bCnl2*weight2)
    gCnl = (gCnl1*weight1+gCnl2*weight2)
    rCnl = (rCnl1*weight1+rCnl2*weight2)
    show2(bCnl[0]*255, "bCnl", False)
    show2(gCnl[0]*255, "gCnl", False)
    show2(rCnl[0]*255, "rCnl", False)

    lb = [bCnl, gCnl, rCnl]

    bChannel = IDreconstructLaplacianPyramid(bCnl)
    gChannel = IDreconstructLaplacianPyramid(gCnl)
    rChannel = IDreconstructLaplacianPyramid(rCnl)

    show2(bChannel, "bChannel", False)
    show2(gChannel, "gChannel", False)
    show2(rChannel, "rChannel", False)
    cv2.waitKey(0)
    fusion = cv2.merge((bChannel, gChannel, rChannel))
    show2(fusion, "fusion", False)
    cv2.waitKey(0)
    return fusion


def normalize(img):
    for i in range(len(img)):
        extractCond = img[i] > 255
        img[i] = np.where(extractCond, 255, img[i])
        extractCond = img[i] < 0
        img[i] = np.where(extractCond, 0, img[i])
    return img
###############################################
# Airlight Estiamte
# DONE but not sure
###############################################


def AEestimate(img, blockSize):
    img = np.array(img)
    rows = img.shape[0]
    cols = img.shape[1]
    while rows*cols > blockSize:
        midRow = int(np.floor(rows/2))
        midCol = int(np.floor(cols/2))
        subIm = [0, 1, 2, 3]
        subIm[0] = img[0:midRow, 0:midCol]
        subIm[1] = img[midRow:rows, 0:midCol]
        subIm[2] = img[0:midRow, midCol:cols]
        subIm[3] = img[midRow:rows, midCol:cols]
        score = [0, 1, 2, 3]
        score[0] = AEcalculateScore(subIm[0])
        score[1] = AEcalculateScore(subIm[1])
        score[2] = AEcalculateScore(subIm[2])
        score[3] = AEcalculateScore(subIm[3])
        index = score.index(max(score))
        img = copy.deepcopy(subIm[index])
        rows = img.shape[0]
        cols = img.shape[1]
    index_X, index_Y = 0, 0
    # print(img)
    tmpValue = ((img[0]-255)**2 + (img[1]-255)**2 + (img[2]-255)**2)**.5
    # print(tmpValue)
    tmpValue = np.array(tmpValue)
    pointValue = min(tmpValue.flatten())
    # print(pointValue)
    a = np.argwhere(tmpValue - pointValue < 1)
    x, y = a[0]
    return img[x, y]


def AEcalculateScore(img):
    mean, std = cv2.meanStdDev(img)
    # not sure why they get (0,0)
    score = np.sum(mean[0] - std[0])
    return score
##############################################
# Transmission Estimate
# dont have transEstiamteEachChannel, else complete
###############################################


def TEtransEstimate(img, patchSz, airlight, lamb, fTrans, r, eps, gamma):
    bgr = cv2.split(img)
    rows, cols, type = img.shape[0], img.shape[1], bgr[0].dtype

    T = TEcomputeTrans(img, patchSz, rows, cols, type, airlight, lamb, fTrans)
    # show(T,"T",False) #done checked
    if r != None and eps != None and gamma != None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = np.uint8(img)
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # WHY? AGAIN? this uses 8UC1
        gray = np.float32(gray)
        T = FguidedImageFilter(gray, T, r, eps)
        Tsmooth = cv2.GaussianBlur(T, (81, 81), 40)
        Tdetails = np.subtract(T, Tsmooth)

        Tdetails = np.multiply(Tdetails, gamma)
        T = Tsmooth + Tdetails
    #show2(Tsmooth,"T smooth",False)
    #show2(Tdetails,"T details",False)
    return T


def TEcomputeTrans(img, patchSz, rows, cols, type, airlight, lamb, fTrans):
    global la, le
    la = []
    ld = True
    le = []
    T = np.zeros((rows, cols), dtype=type)
    for i in range(0, rows, patchSz):
        for j in range(0, cols, patchSz):
            endRow = min(i+patchSz, rows)
            endCol = min(j+patchSz, cols)
            blkIm = img[i:endRow, j:endCol]
            # done, checked.
            Trans = BTEblkEstimate(blkIm, airlight, lamb, fTrans)
            la.append(Trans)
            for m in range(i, endRow):
                for n in range(j, endCol):
                    T[m, n] = Trans

    return T


def LA(self):
    global la, lb, lc, le, lf, lg
    return [la, lb, lc, le, lf, lg]
###########################################
# Blk Transmission Estimate
# dont have each channel version
############################################


def BTEblkEstimate(blkIm, airlight, lamb, fTrans):
    global ld
    le = []
    Trans = 0.0
    nTrans = np.floor(1.0/fTrans*128)
    fMinCost = sys.maxsize
    # or 9223372036854775807
    numberOfPixels = blkIm.shape[0]*blkIm.shape[1]*blkIm.shape[2]
    nCounter = 0
    bgr = cv2.split(blkIm)
    while(nCounter < (1-fTrans)*10):
        color = copy.deepcopy(bgr[0])
        bChannel = BTEpreDehaze(color, airlight[0], nTrans)
        #print("bChannel "+str(bChannel[0,0]))
        if ld:
            ld = False
            le = bChannel
            guess = False

        color = copy.deepcopy(bgr[1])
        gChannel = BTEpreDehaze(color, airlight[1], nTrans)

        color = copy.deepcopy(bgr[2])
        rChannel = BTEpreDehaze(color, airlight[2], nTrans)

        nSumOfLoss = 0

        condition = bChannel > 255
        condExtract = np.array(np.extract(condition, bChannel))
        condExtract = np.sum((condExtract-255)**2)
        nSumOfLoss += condExtract
        condition = bChannel < 0
        condExtract = np.array(np.extract(condition, bChannel))
        nSumOfLoss += np.sum(condExtract**2)

        condition = gChannel > 255
        condExtract = np.array(np.extract(condition, gChannel))
        condExtract = np.sum((condExtract-255)**2)
        nSumOfLoss += condExtract
        condition = gChannel < 0
        condExtract = np.array(np.extract(condition, gChannel))
        nSumOfLoss += np.sum(condExtract**2)

        condition = rChannel > 255
        condExtract = np.array(np.extract(condition, rChannel))
        condExtract = np.sum((condExtract-255)**2)
        condition = rChannel < 0
        condExtract = np.array(np.extract(condition, rChannel))
        nSumOfLoss += np.sum(condExtract**2)

        nSumOfSquareOuts = np.sum(np.multiply(bChannel, bChannel))+np.sum(
            np.multiply(gChannel, gChannel))+np.sum(np.multiply(rChannel, rChannel))
        nSumOfOuts = np.sum(bChannel)+np.sum(gChannel)+np.sum(rChannel)
        fMean = nSumOfOuts/numberOfPixels
        fCost = lamb * nSumOfLoss / numberOfPixels - \
            (nSumOfSquareOuts/numberOfPixels - fMean**2)
        if nCounter == 0 or fMinCost > fCost:
            fMinCost = fCost
            Trans = fTrans
        fTrans = fTrans + 0.1
        nTrans = 1.0/fTrans*128.0
        nCounter = nCounter+1
    return Trans


def BTEpreDehaze(img, a, Trans):
    img = np.array(img)
    return ((img-a)*Trans+128*a)/128

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


def FguidedImageFilter(I, p, r, eps):
    I = np.array(I, np.float64)
    #p = cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
    p = np.array(p, np.float64)

    N = cv2.boxFilter(
        np.ones((I.shape[0], I.shape[1]), dtype=I[0].dtype), -1, (r, r))
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(np.multiply(I, p), cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - np.multiply(mean_I, mean_p)
    mean_II = cv2.boxFilter(np.multiply(I, I), cv2.CV_64F, (r, r))
    var_I = mean_II - np.multiply(mean_I, mean_I)
    a = var_I + eps
    a = cov_Ip/a
    b = mean_p - np.multiply(a, mean_I)
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = np.multiply(mean_a, I)+mean_b
    q = np.float32(q)
    return q


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


def Fapply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def Fapply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = Fapply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = Fapply_mask(matrix, high_mask, high_value)

    return matrix


def Fsimple_ColorBalance(img, percent):
    assert img.shape[2] == 3
    assert 0 <= percent <= 100

    half_percent = percent / 200

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape[:2]
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        low_val = np.percentile(flat, half_percent * 100)
        high_val = np.percentile(flat, (1 - half_percent) * 100)

        # saturate below the low percentile and above the high percentile
        thresholded = Fapply_threshold(channel, low_val, high_val)

        # scale the channel
        normalized = cv2.normalize(
            thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

######################################
# Feature Weight
#######################################


def FWsaliency(img):
    gfbgr = cv2.GaussianBlur(img, (3, 3), 3)
    LabIm = cv2.cvtColor(gfbgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(LabIm)
    l, a, b = np.array(l, dtype=np.float32), np.array(
        a, np.float32), np.array(b, np.float32)
    lm, am, bm = np.mean(l), np.mean(a), np.mean(b)
    l, a, b = l-lm, a-am, b-bm
    sm = np.multiply(l, l)+np.multiply(a, a)+np.multiply(b, b)
    return sm


def FWexposedness(img):
    sigma = 0.25
    average = .5
    img = np.array(img)
    exposedness = np.exp(-1 * (img-average)**2/sigma**2)
    return exposedness


def FWluminanceWeight(img, L):
    bCnl, gCnl, rCnl = cv2.split(img)
    bCnl = np.float32(bCnl)
    gCnl = np.float32(gCnl)
    rCnl = np.float32(rCnl)
    '''
    for i in range(len(L)):
	    for j in range(len(L[0])):
    		part1 = (bCnl[i][j]/255.0 - L[i][j])**2
    		part2 = (gCnl[i][j]/255.0 - L[i][j])**2
                part3 = (rCnl[i][j]/255.0 - L[i][j])**2
	    	lum[i][j]
    '''

    lum =(1+(bCnl/255 -L)**2 + (gCnl/255 -L)**2 + (rCnl/255 - L)**2)**0.5

    return lum

def main():
    test(sys.argv[1],int(sys.argv[2]))


if __name__ == "__main__":
    main()
