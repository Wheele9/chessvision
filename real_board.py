import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter
import time 

def centre_of_4_points(p1,p2,p3,p4):
    return int((p1[0]+p2[0]+p3[0]+p4[0])/4), int((p1[1]+p2[1]+p3[1]+p4[1])/4)


filename = 'reka_1.jpg'
img=cv2.imread(filename)
imi=img
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners

gray = cv2.GaussianBlur(gray,(5,5),0)
threshold = cv2.adaptiveThreshold(gray,255,1,1,11,2)



im2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggest = None
max_area = 0

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


#print (biggest)
#ii=1
#for corns in biggest:
#	print (corns)
#	cv2.circle(img,(corns[0][0],corns[0][1]), 3*ii, (0,0,255), -1)
#	ii=ii*2
##cv2.imshow('masked_treshold',img)

###crating binary image from border
mask = np.ones(img.shape,np.uint8)
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
cv2.fillPoly(mask, pts =[m_c], color=(255,255,255))

ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

if ret == True:
    img = cv2.drawChessboardCorners(mask, (7,7), corners,ret)


    cv2.imshow('img',img)
    #print (3)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

##In the pointPolygonTest function, third argument is measureDist. If it is True, it finds the signed distance. 
##If False, it finds whether the point is inside or outside or on the contour 
##(it returns +1, -1, 0 respectively).

dist = cv2.pointPolygonTest(m_c,tuple(corners[4][0]),True)  # False make it faster return
distances=[cv2.pointPolygonTest(m_c, tuple(point[0]),False) for point in corners]

## if all points are within the contour
if all(p == 1.0 for p in distances): 
	print ('yeaah')



biggest=np.vstack(biggest).squeeze()
m_c = np.vstack(m_c).squeeze()
corners = np.vstack(corners).squeeze()
#print ('dist: ',dist)


CLR = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

right_edge = []
for i in range(7): # 7 lines per border line
	y_coord=int(corners[i*7][1])
	x_coord =int(2*corners[i*7][0]-corners[i*7+1][0])
	right_edge.append([x_coord,y_coord])
	cv2.circle(CLR,(x_coord,y_coord), 10, (0,0,255), -1)

left_edge = []
for i in range(7): # 7 lines per border line
	y_coord=int(corners[i*7+6][1])
	x_coord =int(2*corners[i*7+6][0]-corners[i*7+5][0])
	left_edge.append([x_coord,y_coord])
	cv2.circle(CLR,(x_coord,y_coord), 10, (0,0,255), -1)

bottom_edge = []
for i in range(7): # 7 lines per border line
	x_coord=int(corners[i][0])
	y_coord =int(2*corners[i][1]-corners[i+7][1])
	bottom_edge.append([x_coord,y_coord])
	cv2.circle(CLR,(x_coord,y_coord), 10, (0,0,255), -1)

top_edge = []
for i in range(7): # 7 lines per border line
	x_coord=int(corners[i+42][0])
	y_coord =int(2*corners[i+42][1]-corners[i+35][1])
	top_edge.append([x_coord,y_coord])
	cv2.circle(CLR,(x_coord,y_coord), 10, (0,0,255), -1)



#cv2.imshow('after_intersect_search',CLR)

###TODO create list from corners, edges, 4corners
###tr, tl bl br
#print ("corners: ", corners)
#print ("biggest: ", biggest)
#print ("top_edge: ", top_edge)


proper_corners=[]

proper_corners.extend([biggest[3]])
proper_corners.extend(bottom_edge)
proper_corners.extend([biggest[2]])

for i in range(7):
	proper_corners.extend([right_edge[i]])
	proper_corners.extend(corners[i*7:i*7+7])
	proper_corners.extend([left_edge[i]])

proper_corners.extend([biggest[0]])
proper_corners.extend(top_edge)
proper_corners.extend([biggest[1]])


font = cv2.FONT_HERSHEY_SIMPLEX
letters=['h','g','f','e','d','c','b','a']
numbers=['1','2','3','4','5','6','7','8']
numbersinv=['8','7','6','5','4','3','2','1']
square_names=list(itertools.product(numbers,letters))

square_names=[ss[::-1] for ss in square_names]

#print (square_names)

for cornerpoint in proper_corners:
	pass
	#print (2,cornerpoint)
	cv2.circle(imi,(cornerpoint[0],cornerpoint[1]), 10, (0,0,255), -1)
	
points=proper_corners
for i in range(0,72):
    if (i-8)%9!=0 :
        m_i=i-int(i/9)
        coa=centre_of_4_points(points[i],points[i+1],points[i+9],points[i+10])

        cv2.putText(imi,str(square_names[m_i][0]+square_names[m_i][1]),(coa[0],coa[1]), font, 0.91,(60,230,90),2,cv2.LINE_AA)
        #print (i,m_i, str(square_names[m_i][0]+square_names[m_i][1]))



cv2.imshow("all the points, " , imi)

if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()

cv2.imwrite('real_chessboard.png',imi)
