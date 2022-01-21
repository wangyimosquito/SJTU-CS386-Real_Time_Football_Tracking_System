import numpy as np
import cv2

playerrec_filepath = '..//txt//playerrec.txt'
teamhue_filepath = '..//txt//teamhue.txt'
video_filepath = '..//vid//panorama.avi'
playerCount = 0
playerRec = np.empty([20,4], dtype = "int32")
select_flag = False
myRec =np.zeros([4],dtype="int32")
origin = np.zeros([2],dtype="int32")

'''对于第一帧，可以让用户手动标注所有球员位置，手动选区拉框，后续计算根据这个选区来用卡尔曼滤波预测'''

def MouseRecPicking(event, x, y, flag, param):
    global playerCount
    global select_flag
    global myRec
    global origin
    if(select_flag == True):
        myRec[0] = min(origin[0],x)
        myRec[1] = min(origin[1],y)
        myRec[2] = abs(x-origin[0])
        myRec[3] = abs(y-origin[1])
    if (event == cv2.EVENT_LBUTTONDOWN):
        select_flag = True
        origin = [x,y]
        myRec = [x,y,0,0]
    elif (event == cv2.EVENT_LBUTTONUP):
        select_flag = False
        if(playerCount<20):
            playerRec[playerCount,:] = myRec
        else:
            print("press key for continue")
        playerCount += 1




def firstframe():
    vid_cap = cv2.VideoCapture(video_filepath)
    _, FirstFrame = vid_cap.read()
    originalFrame = FirstFrame
    ####################用户手选取矩形框#######################
    print("processing the first frame")
    print ("Select all the palyers from the Background")
    print ("there should be 20 players in the ground.")
    cv2.namedWindow('First_Frmae',cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('First_Frmae', MouseRecPicking, None)
    for i in range(20):
        cv2.imshow('First_Frmae', FirstFrame)
        cv2.waitKey(0)
        cv2.rectangle(FirstFrame, (myRec[0],myRec[1]), (myRec[0]+myRec[2],myRec[1]+myRec[3]),(0,0,255),1)
    print(playerRec)
    np.savetxt(playerrec_filepath, playerRec)

def avgteamhue():
    #################提取两足球队平局色调#####################
    vid_cap = cv2.VideoCapture(video_filepath)
    _, originalFrame = vid_cap.read()
    hsv = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2HSV)
    sum = 0
    n_points = 0
    playerRec = np.loadtxt(playerrec_filepath)
    print(playerRec)
    for i in range(10):
        for j in range(1):
            for k in range(1):
                width = playerRec[i][2]
                height = playerRec[i][3]
                y = k+playerRec[i][1]+int(height)
                x = j+playerRec[i][0]+int(width)
                sum += hsv[int(y),int(x),0]
                print(hsv[int(y),int(x),0],hsv[int(y),int(x),1],hsv[int(y),int(x),2])
                n_points += 1
    t1hue = sum//n_points

    sum = 0
    n_points =0
    for i in range(10,20):
        for j in range(1):
            for k in range(1):
                width = playerRec[i][2]
                height = playerRec[i][3]
                y = k+playerRec[i][1]+int(height)
                x = j+playerRec[i][0]+int(width)
                sum += hsv[int(y), int(x), 0]
                print(hsv[int(y),int(x),0],hsv[int(y),int(x),1],hsv[int(y),int(x),2])
                n_points += 1
    t2hue = sum//n_points
    print(t1hue,t2hue)
    np.savetxt(teamhue_filepath,[t1hue,t2hue])
#firstframe()
#avgteamhue()

