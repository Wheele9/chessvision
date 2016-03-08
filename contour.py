import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from operator import itemgetter


a = ["foo", "melon", "ddssf",64,532,0,36]
b = [True, False]
c = list(itertools.product(a,b))

print (c)