import cv2
import numpy as np
counter = 0
field_corners = np.empty([4,2], dtype = "float32")

'''
手动标记足球场的顶点，和获取坐标值
'''

def click(event, x, y, flags, param):
    videoFile = '../vid/panorama.avi'
    vid_cap = cv2.VideoCapture(videoFile)
    if vid_cap.isOpened():
        _,first_frame = vid_cap.read()
        first_frame = cv2.resize(first_frame, (2000,600))
        hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

        global counter
        if (event == cv2.EVENT_LBUTTONUP):
            if (counter >=4):
                print ("Press any key to continue")
            else:
                field_corners[counter,:] = [x,y]
                print(x,y)
                print("hue: ",hsv[y,x,0])
                counter +=1

def getground():
    print ("Select the four corners from the Background")
    print ("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
    videoFile = '../vid/panorama.avi'
    vid_cap = cv2.VideoCapture(videoFile)
    if vid_cap.isOpened():
        _,first_frame = vid_cap.read()
        first_frame = cv2.resize(first_frame, (2000,600))
        hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
        cv2.namedWindow('Top')
        cv2.setMouseCallback('Top', click, None)
        cv2.imshow('Top', first_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#getground()

#top view ground corner coordinate (42,34) (42,392) (598,34) (598,392)