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

def rectify(h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
 
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
         
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
  
        return hnew


iter1=0
filename = 'green_board_5.jpg'
img=cv2.imread(filename)
orig = cv2.imread(filename)
gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
print (gray.shape)
gray = cv2.GaussianBlur(gray,(5,5),0)
threshold = cv2.adaptiveThreshold(gray,255,1,1,11,2)

dilated = cv2.dilate(threshold, np.ones((3, 3)))

kernel = np.ones((2,2),np.uint8)
erosion = cv2.erode(threshold,kernel,iterations = 1)

#cv2.imshow('threshold',threshold)
#cv2.imshow('dilated',dilated)
#cv2.imshow('erosion',erosion)
###todo morphology
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
im2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggest = None
max_area = 0
bigest=0

min_size = threshold.size/4
for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                #Approximates a polygonal curve(s) with the specified precision

                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area
                        m_c = i # main countur


_4_corners=rectify(biggest)


###drawing the board border
#cv2.drawContours(orig, [m_c], 0, (0,0,255), 4)



###crating binary image from border
mask = np.ones(orig.shape,np.uint8)
#mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
cv2.fillPoly(mask, pts =[m_c], color=(255,255,255))


threshold = cv2.adaptiveThreshold(gray,255,1,1,11,2)
masked_th=cv2.bitwise_and(threshold,mask)
cv2.imshow('masked_treshold',masked_th)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('real_board.png',orig)
