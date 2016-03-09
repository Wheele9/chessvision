import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter

def nothing(x):
    pass

def centre_of_4_points(p1,p2,p3,p4):
    
    return int((p1[0]+p2[0]+p3[0]+p4[0])/4), int((p1[1]+p2[1]+p3[1]+p4[1])/4)

def absol2D(x,y):

    return x*x+y*y

def intersectionS(line1,line2):

    r1=line1[0]
    t1=line1[1]
    r2=line2[0]
    t2=line2[1]

    det=np.cos(t1)*np.sin(t2)-np.sin(t1)*np.cos(t2)
    #print("determinant: ",det)
    if det==0:
        # line are paralel
        return False
    else:
        global px
        global py
        px=int((np.sin(t2)*r1-np.sin(t1)*r2)/det)
        py=-int((np.cos(t2)*r1-np.cos(t1)*r2)/det)  #image coords...
        if px>0 and px< width and py>0 and py< height:
            #print(px,py)
            cv2.circle(lineimg,(px,py), 4, (255,0,0), -1)
            #cv2.imshow('after_intersect_search',lineimg)

            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()
            return True
        else: 
            # intersecton out of image
            return False

def dist_of_2_points(p1,p2):
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    return (x2-x1)**2+(y2-y1)**2

iter1=0
filename = 'green_board_5.jpg'
orig = cv2.imread(filename)
gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)




cv2.imshow('image',orig)
cv2.imshow('gray',gray)
cv2.imshow('thresh',thresh)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('real_board.png',thresh)
