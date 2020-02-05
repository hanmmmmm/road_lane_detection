import cv2 
import numpy as np 
#import matplotlib.image as mtimg
#import matplotlib.pyplot as plt
from math import sqrt 


cap = cv2.VideoCapture('online.mp4')


while(cap.isOpened()):
	height = 520 # resolution of video
	width = 922


	ret, frame = cap.read()
	# [B G R] = frame[y,x]
	rows,cols,ch = frame.shape
	
	pts1 = np.float32([[380,300],[570,300],[30,520],[810,520]])
	pts2 = np.float32([[0,0],[922,0],[0,520],[922,520]])

	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(frame,M,(cols,rows))


	hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

	lower_yellow = np.array([20,50,90])
	upper_yellow = np.array([30,255,180])

	lower_white = np.array([60,0,90])
	upper_white = np.array([120,30,130])

	mask_w = cv2.inRange(hsv, lower_white, upper_white)
	mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
	res_w = cv2.bitwise_and(dst,dst, mask= mask_w)
	res_y = cv2.bitwise_and(dst,dst, mask= mask_y)

	# cut warpped image into 12 parts
	# 520 / 6 ~= 86
	# 922 / 2  = 461
	sub1l = res_y[0:86,   0:461].copy()
	sub1r = res_w[0:86,   462:922].copy()
	#sub1l = res_y[0:26,   0:461].copy()
	#sub1r = res_w[0:26,   462:922].copy()
	sub2l = res_y[86:172, 0:461].copy()
	sub2r = res_w[86:172, 462:922].copy()
	sub3l = res_y[172:258,0:461].copy()
	sub3r = res_w[172:258,462:922].copy()
	sub4l = res_y[258:344,0:461].copy()
	sub4r = res_w[258:344,462:922].copy()
	sub5l = res_y[344:430,0:461].copy()
	sub5r = res_w[344:430,462:922].copy()
	sub6l = res_y[430:520,0:461].copy()
	sub6r = res_w[430:520,462:922].copy()
	#print(sub4l[50,230])

	rows,cols,cs = sub1l.shape
	sub1l = cv2.flip(sub1l,1)
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub1l[i,j,0] != 0:
				edges[i] = cols-j
				break
		else:
			continue
	sub1y = int(np.mean(edges))
	#print(edges)


	rows,cols,cs = sub1r.shape
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub1r[i,j,0] != 0:
				edges[i] = j
				break
		else:
			continue
	sub1w = int(np.mean(edges))

	


	rows,cols,cs = sub2l.shape
	sub2l = cv2.flip(sub2l,1)
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub2l[i,j,0] != 0:
				edges[i] = cols-j
				break
		else:
			continue
	sub2y = int(np.mean(edges))
	#print(edges)


	rows,cols,cs = sub2r.shape
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub2r[i,j,0] != 0:
				edges[i] = j
				break
		else:
			continue
	sub2w = int(np.mean(edges))

	rows,cols,cs = sub3l.shape
	sub3l = cv2.flip(sub3l,1)
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub3l[i,j,0] != 0:
				edges[i] = cols-j
				break
		else:
			continue
	sub3y = int(np.mean(edges))
	#print(edges)


	rows,cols,cs = sub3r.shape
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub3r[i,j,0] != 0:
				edges[i] = j
				break
		else:
			continue
	sub3w = int(np.mean(edges))

	rows,cols,cs = sub4l.shape
	sub4l = cv2.flip(sub4l,1)
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub4l[i,j,0] != 0:
				edges[i] = cols-j
				break
		else:
			continue
	sub4y = int(np.mean(edges))
	#print(edges)


	rows,cols,cs = sub4r.shape
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub4r[i,j,0] != 0:
				edges[i] = j
				break
		else:
			continue
	sub4w = int(np.mean(edges))

	rows,cols,cs = sub5l.shape
	sub5l = cv2.flip(sub5l,1)
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub5l[i,j,0] != 0:
				edges[i] = cols-j
				break
		else:
			continue
	sub5y = int(np.mean(edges))
	#print(edges)


	rows,cols,cs = sub5r.shape
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub5r[i,j,0] != 0:
				edges[i] = j
				break
		else:
			continue
	sub5w = int(np.mean(edges))

	rows,cols,cs = sub6l.shape
	sub6l = cv2.flip(sub6l,1)
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub6l[i,j,0] != 0:
				edges[i] = cols-j
				break
		else:
			continue
	sub6y = int(np.mean(edges))
	#print(edges)


	rows,cols,cs = sub6r.shape
	#edges = np.zeros(rows*2)
	edges = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			if sub6r[i,j,0] != 0:
				edges[i] = j
				break
		else:
			continue
	sub6w = int(np.mean(edges))

	#print(sub1y,sub1w+461,sub2y,sub2w+461,sub3y,sub3w+461,sub4y,sub4w+461,sub5y,sub5w+461,sub6y,sub6w+461)
	#print('\n\n')
	
	ypoints = np.array([(43,sub1y),(43+86,sub2y),(43+86+86,sub3y),(43+86+86+86,sub4y),(43+86+86+86+86,sub5y),(43+86+86+86+86+86,sub6y)])
	yx = ypoints[:,0]
	yy = ypoints[:,1]

	yz = np.polyfit(yx,yy,2)

	wpoints = np.array([(43,sub1w+461),(43+86,sub2w+461),(43+86+86,sub3w+461),(43+86+86+86,sub4w+461),(43+86+86+86+86,sub5w+461),(43+86+86+86+86+86,sub6w+461)])
	wx = wpoints[:,0]
	wy = wpoints[:,1]

	wz = np.polyfit(wx,wy,2)

	print(int(1/yz[0]),int(1/wz[0]))
	dstcopy = dst.copy()
	#cv2.polylines(dstcopy,ypoints,False, (0, 255, 0),thickness=1, lineType=8, shift=0)

	
	cv2.circle(dst,(sub1y,43),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub2y,43+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub3y,43+86+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub4y,43+86+86+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub5y,43+86+86+86+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub6y,43+86+86+86+86+86),3, (0, 255, 0), -1)

	cv2.circle(dst,(sub1w+461,43),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub2w+461,43+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub3w+461,43+86+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub4w+461,43+86+86+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub5w+461,43+86+86+86+86),3, (0, 255, 0), -1)
	cv2.circle(dst,(sub6w+461,43+86+86+86+86+86),3, (0, 255, 0), -1)
	

	# resize the images, otherwise they are too large to see
	dst08 = cv2.resize(dst, (0,0),None, .8, .8)
	#img = cv2.resize(img, (0,0),None, .8, .8)
	frame08 = cv2.resize(frame, (0,0),None, .8, .8)

	'''
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (40,70)
	fontScale              = 1
	fontColor              = (0,255,0)
	lineType               = 2

	cv2.putText(img,'detected lines', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.putText(crop_back_rgb,'region of interest', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.putText(frame,'raw RGB image', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.putText(grayimgrgb,'gray image', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	'''
	# stack images into one window for easier visulization 
	#img_stack2 = np.vstack((crop_back_rgb, img))
	img_stack1 = np.vstack((frame08, dst08))
	#img_stack = np.hstack((img_stack1, img_stack2))

	cv2.imshow('lines',img_stack1)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break




cap.release()

cv2.destroyAllWindows()














































