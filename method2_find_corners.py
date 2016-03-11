import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter

filename = 'green_board_5.jpg'
img=cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

if ret == True:
    img = cv2.drawChessboardCorners(img, (7,7), corners,ret)


    #cv2.imshow('img',img)

    print (1)
    print (corners)
    ## corners are already in order :) 

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('real_chessboard.png',img)
