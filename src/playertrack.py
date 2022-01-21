from ntpath import join
import numpy as np
from numpy.core.numeric import isclose
#from KalmanFilter import kalmantest
import cv2
from firstframe import firstframe
from hueselector import HueSelector
import topview
import huematcher
import matplotlib.path as path
from PyQt5 import QtCore, QtGui, QtWidgets
import time

'''
调用本文件可执行无用户交互界面的球员跟踪和识别，并在运行结束时保存每个球员的原始图像位置坐标以及俯视图像位置坐标信息
在调用本程序前需要生成以下路径显示的所有文件
'''

bg_filpath = '../img/side-view.jpg'
vid_filepath = '../vid/panorama.avi'
writeVedioName = '../vid/offside.avi'
oripoints_filepath = '..//txt//sideviewcorners.txt'
playerrec_fileoath = '..//txt//playerrec.txt'
hue_filepath = '..//txt//hue.txt'
data_filepath = '..//txt//playerpos.txt'
hgmatrix_filepath = '../txt/hgmatrix.txt'


h = np.loadtxt(hgmatrix_filepath)

def cvImgtoQtImg(cvImg,isConvertToGray=False):  # 定义opencv图像转PyQt图像的函数
    if isConvertToGray:
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_Grayscale8)
    else:        
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGBA)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGBA8888 )
        
    return QtImg
	

class Kalman2D():
    def __init__(self, measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32),
        transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32),
        processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03,
		measurementNoiseCov = np.array([[1,0],[0,1]],np.float32)):

        self.kalman = cv2.KalmanFilter(4,2,0)

        self.meas=[] #history of measurements
        self.pred=[] #history of kalman predictions
        self.mp = np.array((2,1), np.float32) # measurement
        self.tp = np.zeros((2,1), np.float32) # tracked / prediction
        self.kalman.measurementNoiseCov = measurementNoiseCov
        self.kalman.measurementMatrix = measurementMatrix
        self.kalman.transitionMatrix = transitionMatrix
        self.kalman.processNoiseCov = processNoiseCov
        self.hue = 0
        self.numberID = 0
        self.ooc = False
        self.oocIndex = -1
        self.w = 0
        self.h = 0
        self.speed = [0,0]
    def update(self,x,y):
        self.speed = [x-self.mp[0], y-self.mp[1]]
        self.mp = np.array([[np.float32(x)],[np.float32(y)]])
        self.meas.append((x,y))
        self.kalman.correct(self.mp)
        self.tp = self.kalman.predict()
        self.pred.append((int(self.tp[0]),int(self.tp[1])))

    def get_estimate(self):
        return self.pred[-1]

class ShowContours():
	def __init__(self, contours = [0,0,0,0], ooc = False, text = ""):
		self.contours = contours
		self.ooc = ooc
		self.text = text

class OOC():
	def __init__(self, contours = [0,0,0,0], playerID = []):
		self.contours = contours
		self.playerID = playerID
		
def track_player(hg_matrix):
	#############################变量初始化#############################
	playerKalman = list()# 球员卡尔曼列表
	Count = 0 #帧数
	firstFrame = True #第一帧判定
	bg_img = cv2.imread(bg_filpath) #背景画面
	gray_bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)#读入的图像转换为灰度图像格式
	vid_cap = cv2.VideoCapture(vid_filepath) #视频
	hueColor = np.loadtxt(hue_filepath)#球员色调列表
	playerrec = np.loadtxt(playerrec_fileoath)#第一帧20名球员框列表
	currOoc = list()#记录遮挡目标信息
	contourChange = 10
	milisecondtime = list()
	minTime = 100000
	maxTime = -1
	#############################开始处理视频#############################
	while True:
		startTime =time.time()
		KalmanPrediction = list()#该帧卡尔曼滤波预测坐标
		HuePrediction = list()#该帧平均色相值
		players_pos = list()#记录该帧需要显示的框
		playerpos = list()#记录非第一帧该帧需要显示的框
		aval, img = vid_cap.read()
		if not aval:
			break

		#############################赛场背景以及形态学处理#############################
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#当前帧转换为灰度图像格式
		bg_delta = cv2.absdiff(gray_bg_img, gray_img)#与背景图像作差分
		threshold = cv2.threshold(bg_delta, 30, 255, cv2.THRESH_BINARY)[1]#差分后的图像二值化
		threshold = cv2.dilate(threshold, None, iterations=3)#对二值化后的图像作膨胀操作
		threshold = cv2.erode(threshold, None,iterations = 2)#腐蚀操作

		#############################限制赛场选区#############################
		oripoints = np.loadtxt(oripoints_filepath)
		oripoints.astype(np.int32)
		points = np.array([[oripoints[0],oripoints[1],oripoints[2],oripoints[3]]],dtype = np.int32)
		im = np.zeros(threshold.shape[:2],dtype="uint8")
		cv2.polylines(im,points,1,255)
		cv2.fillPoly(im,points,255)
		threshold = cv2.bitwise_and(threshold, threshold, mask = im)

		#############################返回轮廓点集#############################
		contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for i in range(len(contours)):
			(x,y,w,h) = cv2.boundingRect(contours[i])
			players_pos.append(ShowContours(contours = [x,y,w,h]))
		#############################识别轮廓所属#############################
		#2颜色判断
		for cn in contours:
			(x, y, w, h) = cv2.boundingRect(cn)
			player_hue = huematcher.average_hue(x+w//2, y+h//2, w, h, img)
			HuePrediction.append(player_hue)
		#1卡尔曼滤波
		if(firstFrame == True):
			#初始化卡尔曼滤波
			for i in range (20):
				playerKalman.append(Kalman2D())
				(x,y,w,h) = (playerrec[i][0],playerrec[i][1],playerrec[i][2],playerrec[i][3])
				for j in range(2): #卡尔曼滤波初始化循环2次初始值坐标
					playerKalman[i].update(playerrec[i][0],playerrec[i][1])
				playerKalman[i].numberID = i
				KalmanPrediction.append((playerrec[i][0],playerrec[i][1]))
				if i <10:
					playerKalman[i].hue = hueColor[0]
					HuePrediction.append(hueColor[0])
				else:
					playerKalman[i].hue = hueColor[1]
					HuePrediction.append(hueColor[1])
			for i in range (len(players_pos)):
				(x, y, w, h) = cv2.boundingRect(contours[i])
				oocCount = 0
				for j in range(20):
					px = playerKalman[j].mp[0]
					py = playerKalman[j].mp[1]
					pw = playerrec[j][2]
					ph = playerrec[j][3]
					if(px+pw//2>x and px+pw//2 < x+w and py+ph//2>y and py+ph//2<y+h):
						oocCount += 1
						playerKalman[j].oocIndex = i
						playerKalman[j].w = w
						playerKalman[j].h = h
						if (players_pos[i].text != ""):
							players_pos[i].text += ","
						players_pos[i].text += str(j)
				if oocCount > 1:
					players_pos[i].ooc = True
					cplayerID =list()
					for j in range(20):
						if(playerKalman[j].oocIndex == i):
							playerKalman[j].ooc == True
							cplayerID.append(j)
					currOoc.append(OOC(contours = [x,y,w,h], playerID = cplayerID))
							
			#显示第一帧识别结果
			for i in range(20):
				if(playerKalman[i].oocIndex == -1):
					players_pos.append(ShowContours(text = str(i),contours = [int(playerrec[i][0]),int(playerrec[i][1]),int(playerrec[i][2]),int(playerrec[i][3])]))
					playerKalman[i].w = playerrec[i][2]
					playerKalman[i].h = playerrec[i][3]
			for i in range(len(players_pos)):
				if(players_pos[i].text != ""):
					(x,y,w,h) = players_pos[i].contours
					cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#显示白色框
					cv2.putText(img, players_pos[i].text, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
			#cv2.imshow("first frame outcome", img)
			
		
		#################################卡尔曼滤波预测#################################
		elif(Count <-1):
			#对于每个球员初始化其所属轮廓
			for i in range(20):
				playerKalman[i].oocIndex = -1
			#对于每个球员都生成其卡尔曼滤波预测坐标
			for i in range(20):
				KalmanPrediction.append(playerKalman[i].get_estimate())
				cv2.putText(img,str(i),KalmanPrediction[-1], cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,0))
			#对于识别出来的所有目标进行归属判断
			for i in range(len(players_pos)):
				(x,y,w,h) = players_pos[i].contours
				oocCount = 0
				for j in range(20):
					px = KalmanPrediction[j][0]
					py = KalmanPrediction[j][0]
					pw = playerKalman[j].w
					ph = playerKalman[j].h
					if(px+pw//2>x-20 and px+pw//2+20 < x+w and py+ph//2>y-20 and py+ph//2<y+h+20):
						playerKalman.update(x,y)
						oocCount += 1
						playerKalman[j].oocIndex = i
						playerKalman[j].w = w
						playerKalman[j].h = h
						if (players_pos[i].text != ""):
							players_pos[i].text += ","
						players_pos[i].text += str(j)
			for i in range(20):
				if(playerKalman[i].oocIndex == -1):
					playerKalman[i].update(int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]))
					players_pos.append(ShowContours(text = str(i),contours = [int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]),int(playerKalman[i].w),int(playerKalman[i].h)]))
			for i in range(len(players_pos)):
				if(players_pos[i].text != ""):
					(x,y,w,h) = players_pos[i].contours
					cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#显示白色框
					cv2.putText(img, players_pos[i].text, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
					center = [x,y]
					playerpos.append([center, players_pos[i].text])
				else:
					cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 0), 1)#显示黑色框
			if(Count == 2):
				cv2.imshow("2 frame", img)
		###############################依据位置线性计算##############################
		#卡尔曼滤波在线性计算15轮以上时进行预测比较准确，可以在15轮之后加大卡尔曼滤波预测Penal的权重
		else:
			#初始化
			ballFlage = False
			for i in range(20):
				playerKalman[i].oocIndex = -1

			for i in range(len(players_pos)):
				(x,y,w,h) = players_pos[i].contours
				if(w<6 and h<6):#球
					if(ballFlage == False):
						players_pos[i].text += "Ball"
					ballFlage = True
					continue
				oocCount = 0
				singleFlag = False
				if(w<=30 and h <=30):
					singleFlag = True
				if(singleFlag == True):#单人线框选择
					bestMatch = -1
					bestScore = 10000
					score = np.zeros(20)
					for j in range(20):
						if(playerKalman[j].oocIndex == -1):
							px = playerKalman[j].mp[0]
							py = playerKalman[j].mp[1]
							pw = playerKalman[j].w
							ph = playerKalman[j].h
							currSpeed = [x-px,y-py]
							penal = 1000
							if(px+pw//2>x-15 and px+pw//2 < x+w+30 and py+ph//2>y-15 and py+ph//2<y+h+30):
								penal = np.sqrt(np.square(px+pw//2-(x+w//2))+np.square(py+h//2-(y+h//2)))
							huepenal = abs(HuePrediction[i]-playerKalman[j].hue)
							KalmanEst = playerKalman[j].get_estimate()
							KalmanPenal = np.sqrt(np.square(KalmanEst[0]-x)+np.square(KalmanEst[1]-y))
							score[j] = penal + np.sqrt(np.square(currSpeed[0]-playerKalman[j].speed[0])+np.square(currSpeed[1]-playerKalman[j].speed[1])) + huepenal + KalmanPenal
							if(score[j] < bestScore):
								bestMatch = j
								bestScore = score[j]
					if(bestScore < 1000):
						playerKalman[bestMatch].update(x,y)
						playerKalman[bestMatch].oocIndex = i
						playerKalman[j].w = w
						playerKalman[j].h = h
						players_pos[i].text += str(bestMatch)
					
				else:#非单人线框
					for j in range(20):
						if(playerKalman[j].oocIndex == -1):
							px = playerKalman[j].mp[0]
							py = playerKalman[j].mp[1]
							pw = playerKalman[j].w
							ph = playerKalman[j].h
							currSpeed = [x-px,y-py]
							penal = 1000
							if(px+pw//2>x-15 and px+pw//2 < x+w+30 and py+ph//2>y-15 and py+ph//2<y+h+30):
								penal = np.sqrt(np.square(px+pw//2-(x+w//2))+np.square(py+h//2-(y+h//2)))
							huepenal = abs(HuePrediction[i]-playerKalman[j].hue)
							KalmanEst = playerKalman[j].get_estimate()
							KalmanPenal = np.sqrt(np.square(KalmanEst[0]-x)+np.square(KalmanEst[1]-y))
							currscore =  penal + np.sqrt(np.square(currSpeed[0]-playerKalman[j].speed[0])+np.square(currSpeed[1]-playerKalman[j].speed[1])) + huepenal + KalmanPenal
							if(Count<=50 and px+pw//2>x-10 and px+pw//2 < x+w+10 and py+ph//2>y-10 and py+ph//2<y+h+10):
								playerKalman[j].update(x,y)
								playerKalman[j].oocIndex = i
								playerKalman[j].w = w
								playerKalman[j].h = h
								if (players_pos[i].text != ""):
									players_pos[i].text += ","
								players_pos[i].text += str(j)
							elif(Count>50 and currscore < 50):#50帧之后卡尔曼滤波打分存在参考价值
								playerKalman[j].update(x,y)
								playerKalman[j].oocIndex = i
								playerKalman[j].w = w
								playerKalman[j].h = h
								if (players_pos[i].text != ""):
									players_pos[i].text += ","
								players_pos[i].text += str(j)
			#对于没有匹配到的球员，用上一帧的位置代替该帧位置（该方法存疑，会导致一个框长期留在同一位置）
			if(Count < 20):
				for i in range(20):
					if(playerKalman[i].oocIndex == -1):
						playerKalman[i].update(int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]))
						players_pos.append(ShowContours(text = str(i),contours = [int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]),int(playerKalman[i].w),int(playerKalman[i].h)]))
			else:
				originallen = len(players_pos)
				for i in range(20):#反向搜索最适应轮廓并加入
					if(playerKalman[i].oocIndex == -1):
						
						px = playerKalman[i].mp[0]
						py = playerKalman[i].mp[1]
						pw = playerKalman[i].w
						ph = playerKalman[i].h
						KalmanEst = playerKalman[i].get_estimate()
						score = np.zeros(len(players_pos))
						bestMatch = -1
						bestScore = 10000
						for j in range(originallen):
							x = players_pos[j].contours[0]
							y = players_pos[j].contours[1]
							w = players_pos[j].contours[2]
							h = players_pos[j].contours[3]
							KalmanPenal = np.sqrt(np.square(KalmanEst[0]-x)+np.square(KalmanEst[1]-y))
							penal = 1000
							if(px+pw//2>x-15 and px+pw//2 < x+w+30 and py+ph//2>y-15 and py+ph//2<y+h+30):
								penal = np.square(px+pw//2-(x+w//2))+np.square(py+h//2-(y+h//2))
							huepenal = 0.3*abs(HuePrediction[j]-playerKalman[i].hue)
							score[j] = np.sqrt(np.square(currSpeed[0]-playerKalman[i].speed[0])+np.square(currSpeed[1]-playerKalman[i].speed[1])) + huepenal + KalmanPenal
							if(score[j] < bestScore):
								bestScore = score[j]
								bestMatch = j
							#if(i==0):
							#	print("contour",(x,y,h,w)," kalmanPenal: ", KalmanPenal," panel: ", penal, " huepenal: ", huepenal, " score: ", score[j])
						if(bestScore < 300):#存在一个比较合适的轮廓加入
							playerKalman[i].update(players_pos[bestMatch].contours[0],players_pos[bestMatch].contours[1])
							playerKalman[i].oocIndex = bestMatch
							playerKalman[i].w = players_pos[bestMatch].contours[2]
							playerKalman[i].h = players_pos[bestMatch].contours[3]
							if(players_pos[bestMatch].text != ""):
								players_pos[bestMatch].text += ","
							players_pos[bestMatch].text += str(i)
						else:#不存在一个比较合适的轮廓加入，则原地不动，不update
							players_pos.append(ShowContours(text = str(i),contours = [int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]),int(playerKalman[i].w),int(playerKalman[i].h)]))

			# for i in range(len(players_pos)):
			# 	(x,y,w,h) = players_pos[i].contours
			# 	if(players_pos[i].text != ""):
			# 		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#显示白色框
			# 		cv2.putText(img, players_pos[i].text, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
			# 		center = [x,y]
			# 		playerpos.append([center, players_pos[i].text])
			# 	else:
			# 		cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 0), 1)#显示黑色框
			# 		playerpos.append([center, players_pos[i].text])
		
			#显示图中检测到物体的平局色相
			# player_flag = 1
			# if(huematcher.is_team1_player(player_hue,hueColor)):
			# 	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)#黄色
			# 	ch_player = 't1'
			# elif(huematcher.is_team2_player(player_hue,hueColor)):
			# 	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)#红色
			# 	ch_player = 't2'
			# elif(huematcher.is_team1_keeper(player_hue,hueColor)):
			# 	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)#蓝色（两名守门员是蓝色）
			# 	ch_player = 'tk1'
			# elif(huematcher.is_team2_keeper(player_hue,hueColor)):#蓝色
			# 	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
			# 	ch_player = 'tk2'
			
			# else:
			# 	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)#白色,显示无法识别标签的对象
			# 	ch_player = 'o'
			# 	player_flag = 0
			# if player_flag == 1:
			# 	players_pos.append([center_coord, ch_player])
			# 	c +=1
		#前20帧不使用卡尔曼滤波，因为结果不稳定
		# if(firstFrame == False):
		# 	for i in range(20):
		# 		min = 0
		# 		score = list()
		# 		for j in range(len(contours)):
		# 			(x, y, w, h) = cv2.boundingRect(contours[j])
		# 			score.append(abs(x - (playerKalman[i]).meas[-1][0]) + abs(y - (playerKalman[i]).meas[-1][1])) 
		# 			if score[-1]< score[min]:
		# 				min = j
		# 		(x, y, w, h) = cv2.boundingRect(contours[min])
		# 		playerKalman[i].update(x,y)
		# 		playerKalman[i].hue = HuePrediction[j]
		# 		#暂时不排除重复
		# 		ch_player = str(playerKalman[i].numberID)
		# 		center = [x,y]
		# 		playerpos.append([center, ch_player])
		# 		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#显示白色框
		# 		cv2.putText(img, ch_player, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号	

		#3开始匹配每个球员
		# for i in range(20):
		# 	currHueColor = playerKalman[i].hue
		# 	score = list()
		# 	min = 0
		# 	#对于预测值坐标点范围内的球员进行匹配，距离最小的并且色相值最接近的
		# 	for j in range(len(contours)):
		# 		(x, y, w, h) = cv2.boundingRect(contours[j])
		# 		score.append((x-KalmanPrediction[i][0])+ (y-KalmanPrediction[i][1])+ abs(HuePrediction[j] -currHueColor))
		# 		if score[-1] < score[min]:
		# 			min = j
		# 	(x, y, w, h) = cv2.boundingRect(contours[min])
		# 	playerKalman[i].update(x,y)
		# 	playerKalman[i].hue = HuePrediction[j]
		# 	#暂时不排除重复
		# 	ch_player = str(playerKalman[i].numberID)
		# 	center = [x,y]
		# 	players_pos.append([center, ch_player])
		# 	cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#显示白色框
		# 	cv2.putText(img, ch_player, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
		#############################top_view赛场生成#############################
		# if(firstFrame == True):
		# 	for i in range(len(players_pos)):
		# 		center = [players_pos[i].contours[0],players_pos[i].contours[1]]
		# 		playerpos.append([center, players_pos[i].text])

		# top_img = topview.create_topview( hg_matrix= hg_matrix, input_pts=playerpos, selectedID= 0 )
		#side_view赛场生成
		#img = drawoffside.draw(img, player_top_points)
		#img = cv2.resize(img,(0,0),fx=1,fy=1)
		########################进行显示################################
		# cv2.imshow("Player detection", img)
		# cv2.imshow("Top image", top_img)
		# cv2.moveWindow("Top image", 0, 300)
		#key = cv2.waitKey(1) & 0xFF
		Count += 1
		firstFrame = False
		currTime = (time.time()-startTime)*1000
		if(currTime < minTime):
			minTime = currTime
		if(currTime > maxTime):
			maxTime = currTime
		milisecondtime.append(currTime)
		if(Count == 2000):
			print(minTime ,maxTime)
			return

			
		
	# playerDistance.compute(first_player_pos, vid_filepath) # Will compute player distance but was not tested and might cause problems
	# Left commented
	vid_cap.release()
	cv2.destroyAllWindows()

	#########################对所有球员的位置坐标进行保存##############################
	for i in range(20):
		poslist = list()
		topposlist = list()
		for j in range(len(playerKalman[i].meas)):
			poslist.append(playerKalman[i].meas[j])
			pts= np.matrix(np.array([playerKalman[i].meas[j][0],playerKalman[i].meas[j][1],1], dtype = "float32"))
			newPoints = np.empty([1,3], dtype = "float32")
			newPoints = hg_matrix*(pts.T)
			x = int(newPoints[0]/float(newPoints[2]))
			y = int(newPoints[1]/float(newPoints[2]))
			topposlist.append([x,y])
		filename = "..//txt//player"+str(i)+".txt"
		filename2 = "..//txt//top_player"+str(i)+".txt"
		np.savetxt(filename,poslist)
		np.savetxt(filename2, topposlist)


# track_player(hg_matrix= h)