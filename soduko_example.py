import cv2
import numpy as np
import itertools


def intersectionS(line1,line2):

    r1=line1[0]
    t1=line1[1]
    r2=line2[0]
    t2=line2[1]

    det=np.cos(t1)*np.sin(t2)-np.sin(t1)*np.cos(t2)
    print("determinant: ",det)
    if det==0:
        # line are paralel
        return False
    else:
        x=int((np.sin(t2)*r1-np.sin(t1)*r2)/det)
        y=-int((np.cos(t2)*r1-np.cos(t1)*r2)/det)  #image coords...
        if x>0 and x< width and y>0 and y< height:
            print(x,y)
            cv2.circle(lineimg,(x,y), 4, (255,0,0), -1)
            return (x,y)
        else: 
            # intersecton out of image
            return False

iter1=0
filename = 'images2.jpg'
orig = cv2.imread(filename)
shape=orig.shape
height, width, _ = orig.shape
print(shape) # height,width,colors
cv2.imshow('original',orig)
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
th2 = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    81,  #blocksize to consider
    36)  #offset

lineimg=orig

edges = cv2.Canny(blurred,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1.5,np.pi/90,100)
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

    cv2.line(lineimg,(x1,y1),(x2,y2),(0,0,255),2)




cv2.imshow('after_blurr',blurred)
cv2.imshow('after_treshold',th2)
cv2.imshow('after_line_search',lineimg)

combies=list(itertools.combinations(lines,2))

for i in combies:

    #print (i[0][0], i[1][0])
    if intersectionS(i[0][0], i[1][0]):
        pass
        iter1=iter1+1

print (iter1)
cv2.imshow('after_intersect_search',lineimg)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('chess.png',lineimg)

