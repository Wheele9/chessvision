import cv2
import numpy as np
import itertools


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
            return True
        else: 
            # intersecton out of image
            return False


#def dist_of_2_points(x1,y1,x2,y2):
#    return (x2-x1)**2+(y2-y1)**2

def dist_of_2_points(p1,p2):
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    return (x2-x1)**2+(y2-y1)**2

iter1=0
filename = 'images2.jpg'
orig = cv2.imread(filename)

height, width, _ = orig.shape # height,width,colors

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
# combination of all the lines to pairs
# something better algorithm??
px=0
py=0
# initial coordinats of crosses

points=[]
for i in combies:

    #print (i[0][0], i[1][0])
    if intersectionS(i[0][0], i[1][0]):

        points.append([px,py])
        
        iter1=iter1+1
        #points
###SEARCHING THE  4 CORNERS
#print (points)
tl_corner=points[0]
br_corner=points[0]
tl_dist=absol2D(tl_corner[0], tl_corner[1])
br_dist=tl_dist

for i in points:
    #print (i, absol2D(i[0], i[1]))
    if absol2D(i[0], i[1]) < tl_dist:
        print ("closer to tl corner")
        tl_corner=[i[0], i[1] ]
        tl_dist= (absol2D(i[0], i[1]))
    elif absol2D(i[0], i[1]) > br_dist:
        print ("closer to br corner")
        br_corner=[i[0], i[1] ]
        br_dist= (absol2D(i[0], i[1]))

points.remove(br_corner)
points.remove(tl_corner)
## DRAW agreen circle at the corners
cv2.circle(lineimg,(tl_corner[0],tl_corner[1]), 10, (0,255,0), -1)
cv2.circle(lineimg,(br_corner[0],br_corner[1]), 10, (0,255,0), -1)
print ("top left corner ",tl_corner)
print ("bottom right corner ",br_corner)


### FIND THE OTHER 2 CORNERS: DISTANCE FROM CORNERS ARE THE BIGGEST
bl_corner=points[0]


dist_sum_bl=0
for i in points:
    if dist_of_2_points(tl_corner,i)+dist_of_2_points(br_corner,i) > dist_sum_bl:
        dist_sum_bl=dist_of_2_points(tl_corner,i)+dist_of_2_points(br_corner,i)
        bl_corner=[i[0], i[1] ]

points.remove(bl_corner)
cv2.circle(lineimg,(bl_corner[0],bl_corner[1]), 10, (0,255,0), -1)
print ("bottom left corner ",bl_corner)

tr_corner=points[0]
diagonaldist=0

for i in points:
    if dist_of_2_points(bl_corner,i)+dist_of_2_points(br_corner,i)>diagonaldist:
        diagonaldist= dist_of_2_points(bl_corner,i)+dist_of_2_points(br_corner,i)
        tr_corner=[i[0], i[1] ]

cv2.circle(lineimg,(tr_corner[0],tr_corner[1]), 10, (255,255,0), -1)
print ("top right corner ",tr_corner)

cv2.imshow('after_intersect_search',lineimg)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('real_board.png',lineimg)

