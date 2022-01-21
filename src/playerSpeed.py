import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

scalex = 105/555
scaley = 68/363
hgmatrix_filepath = '../txt/hgmatrix.txt'
hg_matrix = np.loadtxt(hgmatrix_filepath)

#生成球员跑动总路程和球员跑动速度柱状图
def playerSpeedgenerate(playerKalman, currFrame, Fps):
    speedlist = list()
    distancelist = list()
    maxspeed = 0
    maxdistance = 0
    currSec = currFrame/Fps
    if(currSec == 0):
        return
    if(currFrame%20 == 19):#20帧取一次位移，每帧取位移会造成结果不稳定
        for i in range(20):
            cx = playerKalman[i].mp[0]
            cy = playerKalman[i].mp[1]
            lx = playerKalman[i].meas[-19][0]
            ly = playerKalman[i].meas[-19][1]
            playerKalman[i].totalDistant += np.sqrt(np.square((cx-lx)*scalex)+np.square((cy-ly)*scaley))#单位m
            playerKalman[i].displaySpeed = playerKalman[i].totalDistant/currSec#单位m/sec
            if(int(playerKalman[i].displaySpeed) > playerKalman[maxspeed].displaySpeed):
                maxspeed = i
            if(int(playerKalman[i].totalDistant) > playerKalman[maxdistance].totalDistant):
                maxdistance = i
            speedlist.append(int(playerKalman[i].displaySpeed))
            distancelist.append(int(playerKalman[i].totalDistant))

    if(currFrame%20 == 19):#每一百帧保存一次结果  
        name_list = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
        plt.bar(range(len(speedlist)), speedlist,tick_label=name_list)
        plt.bar(maxspeed, playerKalman[maxspeed].displaySpeed, color='g')
        plt.savefig("speed.png")
        plt.close()
        plt.bar(range(len(distancelist)), distancelist,tick_label=name_list)
        plt.bar(maxdistance, playerKalman[maxdistance].totalDistant, color='g')
        plt.savefig("Distance.png")
        plt.close()
        # gethtml(speedlist, "Speed")
        # gethtml(distancelist, "Distance")
    return playerKalman

#生成球员跑动路线热力图
def RunningHotMap(playerKalman):
    
    if(not os.path.isfile("..//img//hotMapTeam1.jpg")):
        filename_topview = '../img/blackBg2.png'
        img1 = cv2.imread(filename_topview)
    else:
        img1 = cv2.imread("..//img//hotMapTeam1.jpg")
    if(not os.path.isfile("..//img//hotMapTeam2.jpg")):
        filename_topview = '../img/blackBg2.png'
        img2 = cv2.imread(filename_topview)
    else:
        img2 = cv2.imread("..//img//hotMapTeam2.jpg")
    #生成队1跑动热力图
    for i in range(10):
        pts= np.matrix(np.array([playerKalman[i].meas[-1][0],playerKalman[i].meas[-1][1],1], dtype = "float32"))
        newPoints = np.empty([1,3], dtype = "float32")
        newPoints = hg_matrix*(pts.T)
        x = int(newPoints[0]/float(newPoints[2]))
        y = int(newPoints[1]/float(newPoints[2]))
        cv2.circle(img1,(x,y),2,(255,255,255),-1)#添加球员当前轨迹
    for i in range(10,20):
        pts= np.matrix(np.array([playerKalman[i].meas[-1][0],playerKalman[i].meas[-1][1],1], dtype = "float32"))
        newPoints = np.empty([1,3], dtype = "float32")
        newPoints = hg_matrix*(pts.T)
        x = int(newPoints[0]/float(newPoints[2]))
        y = int(newPoints[1]/float(newPoints[2]))
        cv2.circle(img2,(x,y),2,(255,255,255),-1)#添加球员当前轨迹
    cv2.imwrite("..//img//hotMapTeam1.jpg", img1)
    cv2.imwrite("..//img//hotMapTeam2.jpg", img2)
