import cv2 
import numpy as np 
import matplotlib.image as mtimg
import matplotlib.pyplot as plt
from math import sqrt 


# slow down and stop as STOP sigh
#cap = cv2.VideoCapture('/home/mj/cv_lane_tracking/BDDA/training/camera_videos/20.mp4')

# raining driving with two lines on road
#cap = cv2.VideoCapture('/home/mj/cv_lane_tracking/BDDA/training/camera_videos/60.mp4')

# raining driving with two lines on road
cap = cv2.VideoCapture('/home/mj/cv_lane_tracking/BDDA/training/camera_videos/1306.mp4')
#cap = cv2.VideoCapture('online.mp4')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = 3 #img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

lx1, lx2, ly1, ly2 = 0, 0, 0, 0
rx1, rx2, ry1, ry2 = 0, 0, 0, 0

while(cap.isOpened()):
	height = 720
	width = 1280
	roi = np.array([(30,int(height*0.8)),(400,360),(width/2, height/2),(int(width*0.75), int(height*0.8))], np.int32) 	#region_of_interest
	#roi = np.array([(0,height),(350,300),(550, 300),(int(width*0.85), height)], np.int32) 	#region_of_interest
	#print(roi)

	ret, frame = cap.read()

	grayimg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	cannyed = cv2.Canny(grayimg,100,200)
	cropped = region_of_interest(cannyed,np.array([roi], np.int32),)

	lines = cv2.HoughLinesP(cropped,rho=6,theta=np.pi/60,threshold=190,lines=np.array([]),minLineLength=40,maxLineGap=15)

	#print(lines)

	#if lines != None:
	framecp = np.copy(frame)
	lineimg = np.zeros((height,width,3),dtype=np.uint8)

	#print('{} lines found'.format(len(lines)))

	max_left_longest, max_right_longest = 0, 0 

	for line in lines:
		for x1, y1, x2, y2 in line:
			linelength = sqrt((y2-y1)**2 + (x2-x1)**2)
			slope = float(y2-y1)/float(x2-x1)

			if   slope > 0.8: #right lines
				if linelength > max_right_longest:
					max_right_longest = linelength
					rx1, rx2, ry1, ry2 = x1, x2, y1, y2
					
			elif slope < -0.5 : #left lines
				if linelength > max_left_longest:
					max_left_longest = linelength
					lx1, lx2, ly1, ly2 = x1, x2, y1, y2
	
	rslope = float(ry2-ry1)/float(rx2-rx1)
	lslope = float(ly2-ly1)/float(lx2-lx1)
	#print(rslope)
	rxm, rym, lxm,lym = rx1, ry1, lx1,ly1

	rx1 = int(rxm + (height-rym)/rslope)
	ry1 , ly1 = height, height
	ry2 , ly2 = 400 , 400
	rx2 = int(rxm - (rym-ry2)/rslope)

	lx1 = int(lxm + (height-lym)/lslope)
	lx2 = int(lxm - (lym-ry2)/lslope)
			
	cv2.line(lineimg, (rx1, ry1), (rx2, ry2), [0,255,0], thickness=8)
	cv2.line(lineimg, (lx1, ly1), (lx2, ly2), [0,0,255] , thickness=8)

	img = cv2.addWeighted(frame, 0.8, lineimg, 1.0, 0.0)

	cv2.imshow('cropped_image',img)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


cap.release()

cv2.destroyAllWindows()














































