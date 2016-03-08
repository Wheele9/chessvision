import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter


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

#cv2.imshow('original',orig)
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

    #cv2.line(lineimg,(x1,y1),(x2,y2),(0,0,255),2)




###separating lines with Kmeans

clean_lines=[i[0] for i in lines]

theta=[i[1] for i in clean_lines]
z = np.float32(theta)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(z,3,None,criteria,10,flags)

###change z 2 clean_lines later...
pack1=list(itertools.compress(clean_lines, labels==0))
pack2=list(itertools.compress(clean_lines, labels==1))
pack3=list(itertools.compress(clean_lines, labels==2))


#plt.plot(z)
#plt.plot(pack1)
#plt.plot(pack2)
#plt.plot(pack3)
#plt.show()

if len(pack1)==9:
    biggrup=1
    pack2=pack3+pack2
    #print (11)
elif len(pack2)==9:
    biggrup=2
    pack1=pack3+pack1
    #print (22)
    

elif len(pack3)==9:
    biggrup=3
    pack2=pack1+pack2
    pack1=pack3
    #print (33)

print (222, pack2)
####sort the sparated lines, based on rho


sorted_pack1=sorted(pack1, key=itemgetter(0))
plt.plot(pack2)

###doesnt work quite yet

###add 180 degrees to low theta lines
for i in range(len(pack2)):

    if pack2[i][1]<np.pi/2 : pack2[i][1]=pack2[i][1]+np.pi

print (222, pack2)
plt.plot(pack2)
sorted_pack2=sorted(pack2, key=itemgetter(1))
for i in range(len(pack2)):

    if pack2[i][1]>np.pi : pack2[i][1]=pack2[i][1]-np.pi
###TODO
###subtract pi back to drawing


#uc_x,uc_y=[np.cos(x[1]), np.sin(x[1]) for x in pack2]

# uc_x=map((itemgetter(0)), pack2)
# uc_y=map((itemgetter(0)), pack2)
np_pack2 = np.array(pack2)
uc_x=np.cos((np_pack2[:,1]))
uc_y=np.sin((np_pack2[:,1]))


# plt.plot(uc_x, uc_y)
# plt.plot(uc_y)
#plt.show()



for x in range(len(sorted_pack1)):
    print (111,x, sorted_pack1[x])
    theta=sorted_pack1[x][1]
    rho=sorted_pack1[x][0]
    print (rho,theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(lineimg,(x1,y1),(x2,y2),(x*25,0,255),2)
    #cv2.imshow('after_line_search',lineimg)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

for x in range(len(sorted_pack2)):
    print (111,x, sorted_pack2[x])
    theta=sorted_pack2[x][1]
    if theta>np.pi: theta=theta-np.pi
    rho=sorted_pack2[x][0]
    print (rho,theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(lineimg,(x1,y1),(x2,y2),(x*25,0,255),2)
    cv2.imshow('after_line_search',lineimg)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
#cv2.imshow('after_blurr',blurred)
#cv2.imshow('after_treshold',th2)
#cv2.imshow('after_line_search',lineimg)

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


cv2.imshow('after_intersect_search',lineimg)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imwrite('chess.png',lineimg)

