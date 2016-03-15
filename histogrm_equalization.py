import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter

###global histogram equalization to increase contrast
img = cv2.imread('lc_filed.jpg',0)
 

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side

cv2.imshow("before-after", res)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()

### adaptive histogram equalization

img = cv2.imread('tsukuba_l.png',0)
 
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
 
cv2.imshow("before-after", cl1)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()