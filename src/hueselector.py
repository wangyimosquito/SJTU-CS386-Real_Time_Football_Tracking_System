import cv2
import numpy as np
#根据第一帧，选取球员和守门员队服颜色
hueColor = []
hue_counter = 0
hueList = np.empty([4,2], dtype = "float32")

'''
手动选取球场上两队球员和守门员的色调值
'''

def hue_click(event, x, y, flags, param):
	global hue_counter
	if (event == cv2.EVENT_LBUTTONUP):
		if (hue_counter >=4):
			print ("Press any key to continue")
		else:
			hueList[hue_counter, :] = [x,y]
			print (x,y)
			hue_counter +=1

def HueSelector():
    videoFile = '../vid/panorama.avi'
    vid_cap = cv2.VideoCapture(videoFile)
    if vid_cap.isOpened():
        frame_height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(frame_height,frame_width)
        _,first_frame = vid_cap.read()
        first_frame = cv2.resize(first_frame, (2000,600))
        hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

        print ("Select four objects from the First Frame")
        print ("The selecting sequence should be: Team1, Team2, Keeper1, Keeper2")
        cv2.namedWindow('Side-View')
        cv2.setMouseCallback('Side-View', hue_click, None)
        cv2.imshow('Side-View', first_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # for i in range(7):
        #     print(hueList[i])
        for i in range(4):
            x = int(hueList[i][0])
            y = int(hueList[i][1])
            #计算选取周边5*5掩膜下的平均色相值作为识别颜色
            sum = 0
            for i in range(5):
                for j in range(5):
                    sum += hsv[y-2+j,x-2+i,0]
            hueColor.append(int(sum/25))
        
        # print("Get Hue Color:")
        # for i in range(7):
        #     print(hueColor[i])
        np.savetxt('../txt/hue.txt', hueColor)
    else:
        raise IOError("Could not open video.")
