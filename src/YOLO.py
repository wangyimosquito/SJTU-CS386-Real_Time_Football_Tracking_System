import cv2
import numpy as np
import torch 

bg_filpath = '../img/side-view.jpg'
vid_filepath = '../vid/panorama.avi'
oripoints_filepath = '..//txt//sideviewcorners.txt'

'''对于YOLO5的测试，足球场目标过小，蒙版后的运动员只能识别到近处'''

def YOLOtest():
	bg_img = cv2.imread(bg_filpath) #背景画面
	gray_bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)#读入的图像转换为灰度图像格式
	vid_cap = cv2.VideoCapture(vid_filepath) #视频
	aval, img = vid_cap.read()#img 为第一帧视频
	if not aval:
		return
	cv2.imwrite('..//img//extract2.png',img)
	#############################赛场背景以及形态学处理#############################
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#当前帧转换为灰度图像格式
	bg_delta = cv2.absdiff(gray_bg_img, gray_img)#与背景图像作差分
	threshold = cv2.threshold(bg_delta, 30, 255, cv2.THRESH_BINARY)[1]#差分后的图像二值化
	threshold = cv2.dilate(threshold, None, iterations=3)#对二值化后的图像作膨胀操作
	threshold = cv2.erode(threshold, None,iterations = 2)#腐蚀操作
	img_mask = threshold
	
	oripoints = np.loadtxt(oripoints_filepath)
	oripoints.astype(np.int32)
	points = np.array([[oripoints[0],oripoints[1],oripoints[2],oripoints[3]]],dtype = np.int32)
	im = np.zeros(threshold.shape[:2],dtype="uint8")
	cv2.polylines(im,points,1,255)
	cv2.fillPoly(im,points,255)
	threshold = cv2.bitwise_and(threshold, threshold, mask = im)

	threshold = cv2.bitwise_and(img, img, mask = threshold)
	# cv2.imshow("first frame outcome", threshold)
	# cv2.waitKey(5000)
	cv2.imwrite("..//img//extract.png",threshold)
	
	#Model
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

	# Images
	# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

	# Inference
	results = model(threshold)

	# # Results
	# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

#YOLOtest()