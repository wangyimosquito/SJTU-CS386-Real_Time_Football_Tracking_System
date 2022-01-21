import os
import os.path
import numpy as np
import sys
import cv2
import topview
import huematcher
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import Window
import time
from bgextraction import extract_background
from firstframe import firstframe
from hueselector import HueSelector
from imgeProcess2 import ContourtoPlayer, PlayertoContour, drawContour, drawtopView, preProcess, initialPlayers, ContourtoPlayer2, PlayertoContour2
from playerSpeed import RunningHotMap, playerSpeedgenerate
from ClassDefine import PlayerAction, ShowContours

'''
本文件为PyQt生成的用户可交互界面
'''

bg_filpath = '../img/side-view.jpg'
vid_filepath = '../vid/panorama.avi'
writeVedioName = '../vid/offside.avi'
oripoints_filepath = '..//txt//sideviewcorners.txt'
playerrec_fileoath = '..//txt//playerrec.txt'
hue_filepath = '..//txt//hue.txt'
data_filepath = '..//txt//playerpos.txt'
hgmatrix_filepath = '..//txt//hgmatrix.txt'

scalex = 105/555
scaley = 68/363

playerKalman = list()# 球员列表

def cvImgtoQtImg(cvImg,isConvertToGray=False):  # 定义opencv图像转PyQt图像的函数

    if isConvertToGray:
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_Grayscale8)
    else:        
        QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGBA)
        QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGBA8888 )
    return QtImg

#进行速度图像处理的工作线程类
class workThread(QObject):
	to_show_img_signal = QtCore.Signal(int)
	def __init__(self):
		super(workThread, self).__init__()
	def work(self):
		global Count
		global playerKalman
		playerKalman = playerSpeedgenerate(playerKalman, Count, originalFps)
		RunningHotMap(playerKalman)
		self.to_show_img_signal.emit(1)

##进行主界面图像处理的工作线程类
class workThread2(QObject):
	to_show_img_signal2 = 	QtCore.Signal(int)
	def __init__(self):
		super(workThread2, self).__init__()
	def work(self):

		#--------------------------------------变量初始化--------------------------------------
		global playerKalman
		global Count
		global hg_matrix
		global playerrec
		global originalFps
		global QtImg
		global QtImg2
		global fps
		global frame_count
		global selectedPlayer
		global now_min
		global now_sec

		selectedPlayer = 0
		firstFrame = True #第一帧判定
		bg_img = cv2.imread(bg_filpath) #背景画面
		gray_bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)#读入的图像转换为灰度图像格式
		vid_cap = cv2.VideoCapture(vid_filepath) #视频
		frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)#视频总帧数
		originalFps=vid_cap.get(cv2.CAP_PROP_FPS)#原输入视频帧数
		print("original FPS: ",originalFps)
		fps = 24 #显示视频帧数
		hueColor = np.loadtxt(hue_filepath)#球员色调列表
		playerrec = np.loadtxt(playerrec_fileoath)#第一帧20名球员框列表
		hg_matrix = np.loadtxt(hgmatrix_filepath)#视角转换矩阵
		Count = 0
		#--------------------------------------开始处理视频--------------------------------------
		while True:
			HuePrediction = list()#该帧平均色相值
			players_pos = list()#记录该帧需要显示的框
			playerpos = list()#记录非第一帧该帧需要显示的框
			aval, img = vid_cap.read()
			if not aval:
				break
			#当前帧图像前处理
			if(Count == 0):
				cv2.imwrite("..//img//firstframe.png",img)
			threshold = preProcess(img, gray_bg_img)
			#识别出的目标轮廓点集
			contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			for i in range(len(contours)):
				(x,y,w,h) = cv2.boundingRect(contours[i])
				players_pos.append(ShowContours(contours = [x,y,w,h]))

			#获取轮颜色属性
			for cn in contours:
				(x, y, w, h) = cv2.boundingRect(cn)
				player_hue = huematcher.average_hue(x+w//2, y+h//2, w, h, img)
				HuePrediction.append(player_hue)
			#进行球员和轮廓匹配
			if(firstFrame == True):#对第一帧进行初始化
				players_pos, playerKalman = initialPlayers(players_pos ,playerKalman, playerrec, contours, hueColor, HuePrediction)
				#显示第一帧识别结果
				# for i in range(20):
				# 	if(playerKalman[i].oocIndex == -1):
				# 		players_pos.append(ShowContours(text = str(i),contours = [int(playerrec[i][0]),int(playerrec[i][1]),int(playerrec[i][2]),int(playerrec[i][3])]))
				# 		playerKalman[i].w = playerrec[i][2]
				# 		playerKalman[i].h = playerrec[i][3]
				# for i in range(len(players_pos)):
				# 	if(players_pos[i].text != ""):
				# 		(x,y,w,h) = players_pos[i].contours
				# 		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#显示白色框
				# 		cv2.putText(img, players_pos[i].text, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
			else:#非第一帧
				#将每个球员上一轮所属轮廓Index初始化
				for i in range(20):
					playerKalman[i].oocIndex = -1
				#将球员和轮廓进行正向和反向匹配
				players_pos, playerKalman = PlayertoContour(players_pos, playerKalman, HuePrediction, Count)
				players_pos, playerKalman = ContourtoPlayer(players_pos, playerKalman, HuePrediction, Count)
				#生成主视像
				img, playerpos = drawContour(players_pos, playerKalman, selectedPlayer, img, playerpos)
			#生成俯视图像，更新时间
			playerpos, top_img, now_min, now_sec = drawtopView(players_pos, playerKalman ,playerpos, hg_matrix, Count, originalFps, frame_count, selectedPlayer)
			#转换图像格式
			QtImg = cvImgtoQtImg(img) 
			QtImg2 = cvImgtoQtImg(top_img)
			firstFrame = False
			Count += 1
			#发射信号给主界面，进行图像更新
			cv2.waitKey(int(1000/fps)) #休眠一会，确保每秒播放fps帧
			self.to_show_img_signal2.emit(1)
			#实验只采用1/3原视频长度，方便分析
			if(Count == frame_count/3):
				break
		# 完成所有操作后，释放捕获器
		vid_cap.release()

#用户交互界面
class mainwin(QtWidgets.QMainWindow,Window.Ui_MainWindow):
	_startThread = pyqtSignal()#用于启动速度图像处理的线程信号
	_startThread2 = pyqtSignal()#用于启动主图像处理的线程信号
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.selectedPlayer = 0
		self.aternatePlayer = 0
		self.keeper1Action = list()
		self.keeper2Action = list()
		#AddInfo中显示信息的自定义变量格式
		self.model=QtGui.QStandardItemModel(100,4)
		self.model.setHorizontalHeaderLabels(['Player ID','Time.min','Time.sec','Action Type'])
		self.addTable.setModel(self.model)
		self.row = 0
		self.now_min = 0
		self.now_sec = 0
		#速度图像处理线程
		self.thread = QThread()
		self.work_thread = workThread()
		self.work_thread.moveToThread(self.thread)#moveToThread方法把实例化线程移到Thread管理
		self._startThread.connect(self.work_thread.work)# 线程开始执行之前，从相关线程发射信号
		self.work_thread.to_show_img_signal.connect(self.show_img_in_labelme)#接收子线程信号发来的数据
		#主图像处理线程
		self.thread2 = QThread()
		self.work_thread2 = workThread2()
		self.work_thread2.moveToThread(self.thread2)
		self._startThread2.connect(self.work_thread2.work)
		self.work_thread2.to_show_img_signal2.connect(self.show_img_in_mainview)

	def start(self):
		# if self.thread.isRunning():
		# 	return
		self.thread.start()
		self._startThread.emit()

	#显示速度图像和跑动热力图槽函数
	def show_img_in_labelme(self):
		self.SpeedMap.setPixmap(QtGui.QPixmap("speed.png"))
		#self.SpeedMap.setScaledContents(True)
		self.SpeedMap.show()
		self.TotalDistanceMap.setPixmap(QtGui.QPixmap("Distance.png"))
		#self.TotalDistanceMap.setScaledContents(True)
		self.TotalDistanceMap.show()
		if(Count % 3 ==0):
			while(not os.path.isfile("..//img//hotMapTeam1.jpg")):
				pass
			img1 = cv2.imread("..//img//hotMapTeam1.jpg")
			QtImg1 = cvImgtoQtImg(img1) 
			self.team1hotmap.setPixmap(QtGui.QPixmap.fromImage(QtImg1))
			self.team1hotmap.setScaledContents(True)
			while(not os.path.isfile("..//img//hotMapTeam2.jpg")):
				pass
			img2 = cv2.imread("..//img//hotMapTeam2.jpg")
			QtImg2 = cvImgtoQtImg(img2) 
			self.team2hotmap.setPixmap(QtGui.QPixmap.fromImage(QtImg2))
			self.team2hotmap.setScaledContents(True)
		self.thread.quit()
		#self.thread.terminate()

	def start2(self):
		if self.thread2.isRunning():
			return		
		self.thread2.start()#启动线程
		self._startThread2.emit()

	#显示主图像槽函数
	def show_img_in_mainview(self):
		self.now_min = now_min#更新时间
		self.now_sec = now_sec
		global Count
		self.label.setPixmap(QtGui.QPixmap.fromImage(QtImg))
		size = QtImg.size() 
		self.label_2.setPixmap(QtGui.QPixmap.fromImage(QtImg2))
		size2 = QtImg2.size()
		self.label.resize(size)#根据帧大小调整标签大小
		self.label.show() #刷新界面
		self.label_2.resize(size2)
		if(Count% 3 == 0):
			self.label_2.show()#刷新界面
		
		self.progressBar.setValue(int(Count/frame_count*100))#更新进度条
		self.start()
		if(Count == frame_count):
			self.thread2.quit()#播放完毕时结束线程

	#播放
	def playVideo(self): 
		self.start2()
		time.sleep(0.1)
	
	#锁定当前球员
	def SelectedIDChange(self, i):
		global selectedPlayer
		self.selectedPlayer = i
		selectedPlayer = i
	
	#改变当前球员
	def AlternateIDChange(self,i):
		self.aternatePlayer = i
	
	#进行球员交换
	def ApplyButtonClick(self):
		tmp = playerKalman[self.selectedPlayer]
		playerKalman[self.selectedPlayer] = playerKalman[self.aternatePlayer]
		playerKalman[self.aternatePlayer] = tmp 
	
	#球员动作生成
	def Keeper1Save(self):
		self.keeper1Action.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Keepr 1", action= "Save"))
		item0 = QtGui.QStandardItem("Keeper 1")
		item1 = QtGui.QStandardItem(str(self.keeper1Action[-1].min))
		item2 = QtGui.QStandardItem(str(self.keeper1Action[-1].sec))
		item3 = QtGui.QStandardItem("Save")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()

	def Keeper1Catch(self):
		self.keeper1Action.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Keepr 1", action= "Catch"))
		item0 = QtGui.QStandardItem("Keeper 1")
		item1 = QtGui.QStandardItem(str(self.keeper1Action[-1].min))
		item2 = QtGui.QStandardItem(str(self.keeper1Action[-1].sec))
		item3 = QtGui.QStandardItem("Catch")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()

	def Keeper2Save(self):
		self.keeper2Action.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Keepr 2", action= "Save"))
		item0 = QtGui.QStandardItem("Keeper 2")
		item1 = QtGui.QStandardItem(str(self.keeper2Action[-1].min))
		item2 = QtGui.QStandardItem(str(self.keeper2Action[-1].sec))
		item3 = QtGui.QStandardItem("Save")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()

	def Keeper2Catch(self):
		self.keeper2Action.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Keepr 2", action= "Catch"))
		item0 = QtGui.QStandardItem("Keeper 2")
		item1 = QtGui.QStandardItem(str(self.keeper2Action[-1].min))
		item2 = QtGui.QStandardItem(str(self.keeper2Action[-1].sec))
		item3 = QtGui.QStandardItem("Catch")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def Pass(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Pass"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Pass")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def Dribble(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Dribble"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Dribble")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def Shootout(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Shootout"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Shootout")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def Trap(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Trap"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Trap")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def Block(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Block"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Block")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def TackleS(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Successful Tackle"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Successful Tackle")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	def TackleF(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Failed Tackle"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Failed Tackle")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()

	def AreialDuels(self):
		playerKalman[self.selectedPlayer].playerAction.append(PlayerAction(sec= self.now_sec, min = self.now_min, ID = "Player "+str(self.selectedPlayer), action= "Areial Duels"))
		item0 = QtGui.QStandardItem("Player "+str(self.selectedPlayer))
		item1 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].min))
		item2 = QtGui.QStandardItem(str(playerKalman[self.selectedPlayer].playerAction[-1].sec))
		item3 = QtGui.QStandardItem("Areial Duels")
		#设置每个位置的文本值
		self.model.setItem(self.row,0,item0)
		self.model.setItem(self.row,1,item1)
		self.model.setItem(self.row,2,item2)
		self.model.setItem(self.row,3,item3)
		self.row += 1
		self.addTable.setModel(self.model)
		if(self.row > 99):
			self.addTable.clear()
	
	#删除一行球员动作记录
	def ClearData(self):
		self.model.removeRow(self.row)
		self.row -= 1
		self.addTable.setModel(self.model)
	
	#导出球员动作记录
	def ExportData(self):
		saveRecord =list()
		for i in range(len(self.keeper1Action)):
			saveRecord.append([self.keeper1Action[i].playerID, str(self.keeper1Action[i].min)+':'+str(self.keeper1Action[i].sec), self.keeper1Action[i].type])
		for i in range(len(self.keeper2Action)):
			saveRecord.append([self.keeper2Action[i].playerID, str(self.keeper2Action[i].min)+':'+str(self.keeper2Action[i].sec), self.keeper2Action[i].type])
		for i in range(len(playerKalman)):
				for action in playerKalman[i].playerAction:
					saveRecord.append([action.playerID, str(action.min)+':'+str(action.sec), action.type])
		f=open("..//txt//palyerAction.txt","w")
		for i in range(len(saveRecord)):
			f.write(str(saveRecord[i]) + '\n')
		f.close()
	
	#标记球员四个角落
	def MakeFieldCoordinate(self):
		if(not os.path.isfile(hgmatrix_filepath)):
			if(not os.path.exists('../txt')):
				os.mkdir('../txt')
			print ("Homography matrix has not been created.")
			topview.create_homography()
		print("Homography matrix found.")
		hg_matrix = np.loadtxt(hgmatrix_filepath)
	
	#手动框选球员
	def MarkPlayers(self):
		if(not os.path.isfile(playerrec_fileoath)):
			if(not os.path.exists('../txt')):
				os.mkdir('../txt')
			print("Playerrec of objects has not been selected.")
			firstframe()
		print ("Player Rec Found.")

	#球场背景提取
	def ExtractBg(self):
		if(not os.path.isfile(bg_filpath)):
			if(not os.path.exists('../img')):
				os.mkdir('../img')
			print ("Background has not been extracted, will extract.")
			extract_background(vid_filepath)
		bg_img = cv2.imread(bg_filpath)
		print ("Background image found.")
	
	#清除球场坐标，框选和背景提取记录
	def Clear(self):
		os.remove(hgmatrix_filepath)
		os.remove(playerrec_fileoath)
		os.remove(bg_filpath)

	#清除速度图像记录
	def ClearSpeedMap(self):
		os.remove("speed.png")
	
	#清除路程图像记录
	def ClearDistanceMap(self):
		os.remove("Distance.png")

	#球员颜色选定	
	def SelectColor(self):
		if(not os.path.isfile(hue_filepath)):
			if(not os.path.exists('../txt')):
				os.mkdir('../txt')
			print("Hue of objects has not been selected.")
			HueSelector()
		print ("Hue Color Found.")
	
	#退出程序
	def Quit(self):
		self.thread.quit()
		self.thread.exit()
		self.thread2.quit()
		self.thread2.exit()
		time.sleep(0.1)
		if(os.path.isfile("speed.png")):
			os.remove("speed.png")
		if(os.path.isfile("Distance.png")):
			os.remove("Distance.png")
		if(os.path.isfile("..//img//hotMapTeam1.jpg")):
			os.remove("..//img//hotMapTeam1.jpg")
		if(os.path.isfile("..//img//hotMapTeam2.jpg")):
			os.remove("..//img//hotMapTeam2.jpg")
		sys.exit()