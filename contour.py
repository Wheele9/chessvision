import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter


import numpy as np

 
im = cv2.imread('test.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(im, contours, -1, (0,255,0), 3)


while(1):
    cv2.imshow('image',im)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break