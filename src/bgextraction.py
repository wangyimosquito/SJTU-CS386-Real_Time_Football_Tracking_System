import cv2
import numpy as np
bg_filpath = '..//img//side-view.jpg'


'''
提取球场背景
'''

def extract_background(videoFile):
	vid_cap = cv2.VideoCapture(videoFile)
	if vid_cap.isOpened():
		fps = vid_cap.get(cv2.CAP_PROP_FPS)
		frame_height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		frame_width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)

		print ('FPS', fps)
		print ('Frame Height', frame_height)
		print ('Frame Width', frame_width)
		print ('Frame Count', frame_count)
		frame_count = int(frame_count)

		print ("Extracting background")
		_,img = vid_cap.read()
		avg_img = img

		#use simple average of 500 frame for background extraction
		for fr in range(1, 500): #No need to use whole video for average
			_,img = vid_cap.read()
			fr_fl = float(fr)
			avg_img = (fr_fl*avg_img + img)/(fr_fl+1)
			
		print ("Saving background")
		vid_cap.release()
		cv2.imwrite(bg_filpath, avg_img)
	else:
		raise IOError("Could not open video")

#extract_background('../vid/Panorama.avi')