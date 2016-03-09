"""
opencv tutorials
2015-11-27
matyas-czeman
"""
import numpy as np
import cv2



# Load two images
picture = cv2.imread('bg.png')
mask = cv2.imread('mask.png')

cv2.imshow('picture',picture)
cv2.imshow('mask',mask)
print (picture.shape)
print (mask.shape)

newimg=cv2.bitwise_and(picture,mask)



cv2.imshow('after',newimg)








cv2.waitKey(0)
cv2.destroyAllWindows()
