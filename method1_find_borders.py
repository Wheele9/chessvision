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


###crating binary image from border
mask = np.ones(orig.shape,np.uint8)
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
cv2.fillPoly(mask, pts =[m_c], color=(255,255,255))


threshold = cv2.adaptiveThreshold(gray,255,1,1,11,2)
masked_th=cv2.bitwise_and(threshold,mask)
#cv2.imshow('masked_treshold',masked_th)

#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
kernel = np.ones((3,3),np.uint8)
after_morph = cv2.morphologyEx(masked_th, cv2.MORPH_OPEN, kernel, iterations = 1)
cv2.imshow('morphology',after_morph)








cv2.namedWindow('image')
cv2.createTrackbar('votes','image',0,255,nothing)
cv2.createTrackbar('minlen','image',0,255,nothing)
cv2.createTrackbar('gap','image',0,255,nothing)
cv2.createTrackbar('asize','image',0,20,nothing)


while (1):

    votes = cv2.getTrackbarPos('votes','image')
    minlen = cv2.getTrackbarPos('minlen','image')
    gap = cv2.getTrackbarPos('gap','image')
    asize = cv2.getTrackbarPos('asize','image')

    edges = cv2.Canny(after_morph,100,200, L2gradient =True , apertureSize = (asize*2)+3)
    lines = cv2.HoughLines(edges,1,np.pi/180,votes,minlen,gap)

    img2 = cv2.cvtColor(after_morph,cv2.COLOR_GRAY2BGR)




    #print (lines)
    for x in range(len(lines)):
        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('image',img2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break












if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('real_board.png',orig)
