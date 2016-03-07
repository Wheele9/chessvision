import itertools

a = [1,2,3,4]

b=[[3,5],[43,7],[24,1],[64,1]]

combies= list((itertools.combinations(b,2)))

for i in combies:
    print (i)

print (combies)