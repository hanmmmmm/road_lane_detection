import cv2 
import numpy as np 
#import matplotlib.image as mtimg
#import matplotlib.pyplot as plt
from math import sqrt 


cap = cv2.VideoCapture('online.mp4')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

lx1, lx2, ly1, ly2 = 0, 0, 0, 0
rx1, rx2, ry1, ry2 = 0, 0, 0, 0

old1_rs,old2_rs,old3_rs = 0,0,0
old1_ls,old2_ls,old3_ls = 0,0,0
old1_lx1,old2_lx1,old3_lx1 = 0,0,0


while(cap.isOpened()):
	height = 520 # resolution of video
	width = 922

	#region_of_interest
	roi = np.array([(0,height),(350,300),(550, 300),(int(width*0.85), height)], np.int32) 	
	
	ret, frame = cap.read()

	grayimg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	cannyed = cv2.Canny(grayimg,100,200)
	cropped = region_of_interest(cannyed,np.array([roi], np.int32),)

	lines = cv2.HoughLinesP(cropped,rho=6,theta=np.pi/60,threshold=190,lines=np.array([]),minLineLength=40,maxLineGap=15) #tune the parameters 

	framecp = np.copy(frame)
	lineimg = np.zeros((height,width,3),dtype=np.uint8) # for storing deteced lines into image 

	max_left_longest, max_right_longest = 0, 0 

	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				linelength = sqrt((y2-y1)**2 + (x2-x1)**2)
				slope = float(y2-y1)/float(x2-x1)
				# for each line detected, check if it is on left or right
				# then check if it is longer than the previous lines, to find the longest one
				if   slope > 0: #right lines
					if linelength > max_right_longest:
						max_right_longest = linelength
						rx1, rx2, ry1, ry2 = x1, x2, y1, y2
						
				elif slope < 0 : #left lines
					if linelength > max_left_longest:
						max_left_longest = linelength
						lx1, lx2, ly1, ly2 = x1, x2, y1, y2
	rslope = float(ry2-ry1)/float(rx2-rx1)
	lslope = float(ly2-ly1)/float(lx2-lx1)
	
	# low pass filters for slope values 
	p_rslope = np.mean([rslope,old1_rs,old2_rs,old3_rs])
	old3_rs = old2_rs
	old2_rs = old1_rs
	old1_rs = rslope

	p_lslope = np.mean([lslope,old1_ls,old2_ls,old3_ls])
	old3_ls = old2_ls
	old2_ls = old1_ls
	old1_ls = lslope

	# compute the ends' pixel coordinates for right line and left line 
	rxm, rym, lxm,lym = rx1, ry1, lx1,ly1

	rx1 = int(rxm + (height-rym)/p_rslope)
	ry1 , ly1 = height, height
	ry2 , ly2 = 300 , 300
	rx2 = int(rxm - (rym-300)/p_rslope)

	lx1 = int(lxm + (height-lym)/p_lslope)
	lx2 = int(lxm - (lym-300)/p_lslope)
	# add lines into 'lineimg' 			
	cv2.line(lineimg, (rx1, ry1), (rx2, ry2), [0,255,0], thickness=8)
	cv2.line(lineimg, (lx1, ly1), (lx2, ly2), [0,0,255] , thickness=8)

	# combine raw image and lines together 
	img = cv2.addWeighted(frame, 0.8, lineimg, 1.0, 0.0)

	# convert gray 1 channel images to 3 channel images	
	crop_back_rgb = cv2.cvtColor(cropped,cv2.COLOR_GRAY2RGB)
	grayimgrgb = cv2.cvtColor(grayimg,cv2.COLOR_GRAY2RGB)

	# resize the images, otherwise they are too large to see
	crop_back_rgb = cv2.resize(crop_back_rgb, (0,0),None, .8, .8)
	img = cv2.resize(img, (0,0),None, .8, .8)
	frame = cv2.resize(frame, (0,0),None, .8, .8)
	grayimgrgb = cv2.resize(grayimgrgb, (0,0),None, .8, .8)

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (40,70)
	fontScale              = 1
	fontColor              = (0,255,0)
	lineType               = 2

	cv2.putText(img,'detected lines', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.putText(crop_back_rgb,'region of interest', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.putText(frame,'raw RGB image', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.putText(grayimgrgb,'gray image', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

	# stack images into one window for easier visulization 
	img_stack2 = np.vstack((crop_back_rgb, img))
	img_stack1 = np.vstack((frame, grayimgrgb))
	img_stack = np.hstack((img_stack1, img_stack2))
	cv2.imshow(' as',img_stack)
	#cv2.imshow(' as',frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


cap.release()

cv2.destroyAllWindows()














































