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
        return False
    else:
        x=int((np.sin(t2)*r1-np.sin(t1)*r2)/det)
        y=-int((np.cos(t2)*r1-np.cos(t1)*r2)/det)  #image coords...
        print(x,y)
        cv2.circle(lineimg,(x,y), 4, (255,0,0), -1)
        return (x,y)

filename = 'images2.jpg'
orig = cv2.imread(filename)
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


rho=351
theta=0
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))

cv2.line(lineimg,(x1,y1),(x2,y2),(0,255,0),2)
rho=153
theta=1.5708

a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
cv2.line(lineimg,(x1,y1),(x2,y2),(0,255,0),2)

rho=315
theta=0.10472

a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))
    
cv2.line(lineimg,(x1,y1),(x2,y2),(0,255,0),2)


cv2.imshow('after_blurr',blurred)
cv2.imshow('after_treshold',th2)
cv2.imshow('after_line_search',lineimg)

combies=list(itertools.combinations(lines,2))

for i in combies:
    print (200,i)
    print (i[0][0], i[1][0])
    intersectionS(i[0][0], i[1][0])


cv2.imshow('after_intersect_search',lineimg)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('chess.png',lineimg)