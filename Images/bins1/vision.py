#!/usr/bin/python3
import cv2
import random
import numpy as np
import sys
from matplotlib import pyplot as plt
from math import isnan
import os
from sklearn.cluster import KMeans
from random import shuffle
from scipy.stats import linregress


def distance(color1, color2):
    return np.linalg.norm(color1 - color2)

def linreg_convolution(xylw, src, dst):
    """ Linearizes a chunk of an array in place """
    #corrected = np.zeros(arr.shape)
    y_val = []
    x_val = []
    for y in range(xylw[1], xylw[1] + xylw[2]):
        for x in range(xylw[0], xylw[0] + xylw[3]):
            if src[y,x] > 127:
                x_val.append(x)
                y_val.append(y)
    if len(x_val) == 0:
        return
    slope, intercept, r_value, p_value, std_err = linregress(x_val, y_val)
    if isnan(slope) or isnan(intercept):
        return
    for x in range(xylw[0], xylw[0] + xylw[3]):
        dst[min(dst.shape[0]-1, int(slope * x + intercept)), x] = 255 * r_value


def camera_mask(img):
    mask = cv2.imread("mask.png")
    return cv2.addWeighted(img, .7, mask, .3,0)


def denoise(img):
    return cv2.GaussianBlur(img,(5,5),0)


def color_clustering(img):
    original_shape = img.shape
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters = 20, max_iter=10, n_init=3)
    clt.fit_predict(img)
    whites = sorted(clt.cluster_centers_, key=lambda x: (255 ** 3) - distance(x, [255, 255, 255]))[-3:]
    whites = [i.tolist() for i in  whites]
    indices = []
    for idx, val in enumerate(clt.cluster_centers_):
        if val.tolist() in whites:
            indices.append(idx)

    lab = np.zeros(clt.labels_.shape, dtype='int32')
    np.copyto(lab, clt.labels_)
    lab = lab.reshape(original_shape[:-1])

    cp = np.zeros(original_shape, dtype='uint8')
    for val in indices:
        cp[lab == val] = [255, 255, 255]
    return cp


def laplacian(img):
    final = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(final, kernel, iterations = 1)
    return dilated


def erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    eroded = cv2.erode(img, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
    return cv2.dilate(eroded, kernel, iterations = 1)


def bbox(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    throwaway, contours, throwaway_2 = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        rects.append(np.int0(box))

    
    max_ = max(rects, key = lambda x: cv2.contourArea(x))

    return [max_,]


def process(path):
    original = cv2.imread(path)
    masked = camera_mask(original)
    denoised = denoise(masked)
    top_colors = color_clustering(denoised)
    eroded = erode(top_colors)
    
    boundaries = bbox(eroded)

    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    original = cv2.drawContours(original, boundaries, -1, [0, 255, 0], 2)

    plt.imshow(original)
    plt.show()

ol = os.listdir(sys.argv[1])
shuffle(ol)
if os.path.isdir(sys.argv[1]):
    for i in ol:
        if i[-4:] in (".jpg", ".png"):
            print(i)
            process(sys.argv[1] + i)
else:
    process(sys.argv[1])
