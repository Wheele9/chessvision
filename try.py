"""
opencv tutorials
2015-11-27
matyas-czeman
"""
import numpy as np
import cv2



from operator import itemgetter
L=[[0, 1, 'f'], [4, 2, 't'], [9, 4, 'afsd']]
xx=sorted(L, key=itemgetter(1))


print(xx)


a = [ [1,2], [2,9], [20,7] ]
na = np.array(a)
print (na[:,1])


for elem in a:
    print (elem[:1])
    #if elem[:1] < 10: 
      #  elem[:1]=elem[:1]+23

moda=[z[0]+180 for z in a if z[0] < 2  ]

#print (moda)

print (a)

for i in range(len(a)):
    print (i)
    if (a[i][0]) < 20: a[i][0]=a[i][0]+30


print (str('a'+'b'))


