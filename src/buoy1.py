import cv2
import numpy as np
import copy
import sys
import time
from matplotlib import pyplot as plt


class Buoy:
    def openFile(self, name, path1):
        #"/Users/rongk/Downloads/test.jpg"):
        if name == 'd':
            path0 = "/home/dhyang/Desktop/test/"
            path0 = '/media/dhyang/4E10-86F1/images_real/'
        #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/images/training15.png"
        #path = "/Users/rongk/Downloads/Vision-master/Vision-master/RoboticsImages/03.jpg"
        else:
            path0 = "/home/krong/Documents/vision/"
        path2 = ".jpg"
        path = path0+path1+path2
        img = cv2.imread(path)
        print(path)
        img = cv2.resize(img, (640, 480))
        return img

    def binarization(self, gray):
        #ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,-10)
        th2 = cv2.bitwise_not(th2)
        return th2

    def getLines(self, newImg):
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

    def plotLines(self, lineLocs, original):
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

    def segment(self, image):
        mdpt = (int)(image.shape[0]/2)
        striph = 150
        return image[mdpt - striph: mdpt + striph, :]

    def adjust(self, image):
        alphah = 8
        alphas = 8
        alphav = 8

        h, s, v = cv2.split(image)
        new_image = np.zeros(image.shape, image.dtype)
        h1, s1, v1 = cv2.split(new_image)

        maximum = h.mean()
        #maximum = h.min()
        beta = 127-alphah*maximum  # Simple brightness control
        h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

        maximum = s.mean()
        beta = 127-alphas*maximum  # Simple brightness control
        s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

        maximum = v.mean()
        beta = 127-alphav*maximum  # Simple brightness control
        v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

        new_image = cv2.merge([h1, s1, v1])
        return new_image

    def adjust1(self, image):
        alphah = 8
        alphas = 8
        alphav = 8

        h, s, v = cv2.split(image)
        new_image = np.zeros(image.shape, image.dtype)
        h1, s1, v1 = cv2.split(new_image)

        maximum = h.mean()
        #maximum = h.min()
        beta = -alphah*maximum  # Simple brightness control
        h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

        maximum = s.mean()
        beta = -alphas*maximum  # Simple brightness control
        s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

        maximum = v.mean()
        beta = -alphav*maximum  # Simple brightness control
        v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

        new_image = cv2.merge([h1, s1, v1])
        return new_image

    def adjustYUV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        alphah = 1
        alphas = 0.15
        alphav = 0.15

        h, s, v = cv2.split(image)
        new_image = np.zeros(image.shape, image.dtype)
        h1, s1, v1 = cv2.split(new_image)

        maximum = h.mean()
        #maximum = h.min()
        beta = 127-alphah*maximum  # Simple brightness control
        h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

        maximum = s.mean()
        beta = 127-alphas*maximum  # Simple brightness control
        s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

        maximum = v.mean()
        beta = 127-alphav*maximum  # Simple brightness control
        v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)

        new_image = cv2.merge([h1, s1, v1])
        new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)
        return new_image

    def adjustHSV(self, image):
        alphah = 0
        alphas = 4
        alphav = 2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        new_image = np.zeros(image.shape, image.dtype)
        h1, s1, v1 = cv2.split(new_image)

        maximum = h.mean()
        h1 = cv2.convertScaleAbs(h, alpha=alphah, beta=beta)

        maximum = s.mean()
        beta = 127-alphas*maximum  # Simple brightness control
        s1 = cv2.convertScaleAbs(s, alpha=alphas, beta=beta)

        maximum = v.mean()
        beta = 127-alphav*maximum  # Simple brightness control
        v1 = cv2.convertScaleAbs(v, alpha=alphav, beta=beta)
        new_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        new_image = cv2.merge([h, s1, v1])
        return new_image

    def boundingRectangle(self, original, thresh):
        contours, h = cv2.findContours(thresh, 1, 2)
        leeway = 40
        # print("hi")
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        for cnt in np.flip(cntsSorted):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(cnt)
            if area > 100000 or area < 800:
                continue
            if abs(box[3][0]-box[1][0]) > 5*abs(box[3][1]-box[1][1]):
                continue
            if abs(box[3][0]-box[1][0])*5 < abs(box[3][1]-box[1][1]):
                continue
            cv2.drawContours(original, [box], 0, (0, 0, 255))
            center = int((box[3][0]+box[1][0])/2)
            leeway = int(abs(box[3][0]-box[1][0])*0.75)
            # print(box)
            #leeway = 10

            print(area)
            if center-leeway > 0:
                thresh[:, center-leeway:center] = 255
            else:
                thresh[:, 0:center] = 255
            if center+leeway < thresh.shape[1]:
                thresh[:, center:center+leeway] = 255
            else:
                thresh[:, center:] = 255
            xcoord = center
            ycoord = int((box[3][1]+box[1][1])/2)
            return xcoord, ycoord
        return -1, -1

    def fill(self, original, thresh):
        contours, h = cv2.findContours(thresh, 1, 2)
        img = np.ones([original.shape[0], original.shape[1], 3],
                      dtype=np.uint8)*255
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(cnt)
            if area > 100 and area < 10000:
                cv2.drawContours(img, [box], 0, (0, 0, 0),
                                 thickness=cv2.FILLED)
        return img
    ############################################

    def getMask(self, img):
        lower_green = np.array([0, 0, 0])
        upper_green = np.array([255, 255, 240])
        mask = cv2.inRange(img, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        return mask

    def floodfill(self, img):
        im_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = img | im_floodfill_inv
        return im_out

    def mainImg(self, img):
        start_time = time.time()
        original = cv2.resize(img, (640, 320))
        origin = copy.deepcopy(original)

        o1 = original
        o = copy.deepcopy(original)
        o1 = copy.deepcopy(original)
        cv2.imshow("original", o)
        gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobelx = sobelx + np.min(sobelx)
        r = np.max(sobelx) - np.min(sobelx)
        div = 255/r #calculate the normalize divisor
        sobelx = np.uint8(np.round(sobelx * div))
        sobel=self.binarization(sobelx)
        cv2.imshow("test",sobelx)
        cv2.imshow("binarize",sobel)
        img=cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
        for i in range(20):
            sobel = cv2.erode(sobel,np.ones((5,6)),iterations=1)
            sobel = cv2.dilate(sobel, np.ones((4,4)),iterations=1)

        # create the params and deactivate the 3 filters
        params = cv2.SimpleBlobDetector_Params()
        #params.minThreshold = 20
        params.filterByArea = True
        #params.minArea = 10000
        params.maxArea = 100000

        params.filterByInertia = False
        params.filterByConvexity = False
        # detect the blobs
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(sobel)
        keypoints = sorted(keypoints,key = lambda x: -x.size)
        boxes = []
        for i in range(0):
            keyPoint = keypoints[i]
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]
            s = keyPoint.size
            boxes.append([x,y])
        keyPlot = keypoints[0:2]
        im_with_keypoints = cv2.drawKeypoints(original, keyPlot, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        print(boxes)

        #x1,y1 = self.boundingRectangle(original,sobel)
        cv2.imshow("sdjfs", sobel)
        cv2.imshow("slfkj",original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#####################################################


def main():
    a = Buoy()
    img = a.openFile(sys.argv[1], sys.argv[2])
    boxes = a.mainImg(img)
    print(boxes)


if __name__ == "__main__":
    main()
