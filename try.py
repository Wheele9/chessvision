"""
opencv tutorials
2015-11-27
matyas-czeman
"""
import numpy as np
import cv2
def absol2D(x,y):
    return x*x+y*y

cap = cv2.VideoCapture(0)

ll=[[5,3],[32,5],[-53,1],[53,5]]
for i in ll:
    absol=np.absolute(i)
    #print (i)
    print (absol)

print (np.absolute([5,7]))

print (absoliiii(5,7))
