import cv2
import numpy as np

filename = 'reka_1.jpg'
img=cv2.imread(filename)

x=3

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (x,x),None)
print (corners)
if ret == True:
	print (222)
img = cv2.drawChessboardCorners(img, (x,x), corners,ret)
cv2.imshow('chessboardcorners',img)

if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()
