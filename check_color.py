import cv2 
import numpy as np 
#import matplotlib.image as mtimg
#import matplotlib.pyplot as plt
from math import sqrt 


#cap = cv2.VideoCapture('online.mp4')
img = cv2.imread('white.png')
#img = cv2.imread('red.jpg')
print(img[500,625])
cv2.imshow(' as',img)
cv2.waitKey(25000)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv[500,625])










































