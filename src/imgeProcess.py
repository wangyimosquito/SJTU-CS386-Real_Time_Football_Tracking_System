import cv2
import numpy as np
import topview
from ClassDefine import ShowContours, Kalman2D

oripoints_filepath = '..//txt//sideviewcorners.txt'

#每帧图像前处理
def preProcess(img, gray_bg_img):

	#进行线性和形态学处理
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#当前帧转换为灰度图像格式
	bg_delta = cv2.absdiff(gray_bg_img, gray_img)#与背景图像作差分
	threshold = cv2.threshold(bg_delta, 30, 255, cv2.THRESH_BINARY)[1]#差分后的图像二值化
	threshold = cv2.dilate(threshold, None, iterations=3)#对二值化后的图像作膨胀操作
	threshold = cv2.erode(threshold, None,iterations = 2)#腐蚀操作

	#限制赛场选取
	oripoints = np.loadtxt(oripoints_filepath)
	oripoints.astype(np.int32)
	points = np.array([[oripoints[0],oripoints[1],oripoints[2],oripoints[3]]],dtype = np.int32)
	im = np.zeros(threshold.shape[:2],dtype="uint8")
	cv2.polylines(im,points,1,255)
	cv2.fillPoly(im,points,255)
	threshold = cv2.bitwise_and(threshold, threshold, mask = im)

	return threshold

def initialPlayers(players_pos ,playerKalman, playerrec, contours, hueColor, HuePrediction):
	#初始化卡尔曼滤波
	for i in range (20):
		playerKalman.append(Kalman2D())
		(x,y,w,h) = (playerrec[i][0],playerrec[i][1],playerrec[i][2],playerrec[i][3])
		for j in range(2): #卡尔曼滤波初始化循环2次初始值坐标
			playerKalman[i].update(playerrec[i][0],playerrec[i][1])
		playerKalman[i].numberID = i
		if i <10:
			playerKalman[i].hue = hueColor[0]
			HuePrediction.append(hueColor[0])
		else:
			playerKalman[i].hue = hueColor[1]
			HuePrediction.append(hueColor[1])
	for i in range (len(players_pos)):
		(x, y, w, h) = cv2.boundingRect(contours[i])
		for j in range(20):
			px = playerKalman[j].mp[0]
			py = playerKalman[j].mp[1]
			pw = playerrec[j][2]
			ph = playerrec[j][3]
			if(px+pw//2>x and px+pw//2 < x+w and py+ph//2>y and py+ph//2<y+h):
				playerKalman[j].oocIndex = i
				playerKalman[j].w = w
				playerKalman[j].h = h
				if (players_pos[i].text != ""):
					players_pos[i].text += ","
				players_pos[i].text += str(j)		
	return players_pos, playerKalman

#将球员分配给轮廓
def PlayertoContour(players_pos, playerKalman, HuePrediction, Count):
	for i in range(len(players_pos)):
		(x,y,w,h) = players_pos[i].contours
		if(w<6 and h<6):#球或其他过小目标忽略
			continue
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
					huepenal = 0.3*abs(HuePrediction[i]-playerKalman[j].hue)
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
		else:#多人线框
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
						playerKalman[j].update(KalmanEst[0],KalmanEst[1])
						playerKalman[j].oocIndex = i
						playerKalman[j].w = w
						playerKalman[j].h = h
						if (players_pos[i].text != ""):
							players_pos[i].text += ","
						players_pos[i].text += str(j)
	return players_pos, playerKalman

#将轮廓分配给球员
def ContourtoPlayer(players_pos, playerKalman, HuePrediction,Count):
	if(Count < 20):#小于20帧，由于卡尔曼滤波预测还不稳定，因此用上一帧轮廓代替当前帧轮廓
		for i in range(20):
			if(playerKalman[i].oocIndex == -1):
				playerKalman[i].update(int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]))
				players_pos.append(ShowContours(text = str(i),contours = [int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]),int(playerKalman[i].w),int(playerKalman[i].h)]))
	else:#大于20帧时，反向匹配，在所有轮廓中选入最佳轮廓，并加入
		originallen = len(players_pos)
		for i in range(20):
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
					if(w<6 and h<6):#球或过小目标忽略
						continue
					KalmanPenal = np.sqrt(np.square(KalmanEst[0]-x)+np.square(KalmanEst[1]-y))
					penal = 300
					if(px+pw//2>x-15 and px+pw//2 < x+w+30 and py+ph//2>y-15 and py+ph//2<y+h+30):
						penal = np.square(px+pw//2-(x+w//2))+np.square(py+h//2-(y+h//2))
					bonus = 0#对于暂时还未分配识别目标给予正向奖励
					if(players_pos[j].text == ""):
						bonus -= 70
					huepenal = 0.3*abs(HuePrediction[j]-playerKalman[i].hue)
					score[j] = huepenal + KalmanPenal + bonus + penal
					if(score[j] < bestScore):
						bestScore = score[j]
						bestMatch = j
				if(bestScore < 300):#存在合适的轮廓加入
					if(players_pos[bestMatch].text == ""):#单人轮廓框，直接用该轮廓框进行卡尔曼滤波更新
						playerKalman[i].update(players_pos[bestMatch].contours[0],players_pos[bestMatch].contours[1])
					else:#非单人轮廓框，用其卡尔曼滤波值更新
						est = playerKalman[i].get_estimate()
						playerKalman[i].update(est[0],est[1])
					playerKalman[i].oocIndex = bestMatch
					playerKalman[i].w = players_pos[bestMatch].contours[2]
					playerKalman[i].h = players_pos[bestMatch].contours[3]
					if(players_pos[bestMatch].text != ""):
						players_pos[bestMatch].text += ","
					players_pos[bestMatch].text += str(i)
				else:#不存在一个比较合适的轮廓加入，则原地不动，但不update，以免陷入卡死不动的情况
					players_pos.append(ShowContours(text = str(i),contours = [int(playerKalman[i].mp[0]),int(playerKalman[i].mp[1]),int(playerKalman[i].w),int(playerKalman[i].h)]))
	return players_pos, playerKalman

#在主图像中画出轮廓
def drawContour(players_pos, playerKalman, selectedPlayer, img, playerpos):
	for i in range(len(players_pos)):
		(x,y,w,h) = players_pos[i].contours
		center = [x,y]
		if(players_pos[i].text != ""):
			if(playerKalman[selectedPlayer].oocIndex == i):
				cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 1)#对于当前选择的球员，其轮廓为红色
				cv2.putText(img, players_pos[i].text, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,255))#显示球员字号
			#else:
				#cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 1)#其他球员轮廓显示白色
				#cv2.putText(img, players_pos[i].text, (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
		playerpos.append([center, players_pos[i].text])
		# else:
		# 	cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 0), 1)#显示黑色框
		# 	playerpos.append([center, players_pos[i].text])
	return img, playerpos

#在俯视图中画出轮廓
def drawtopView(players_pos, playerKalman ,playerpos, hg_matrix, Count, originalFps, frame_count, selectedPlayer):
	if(Count == 0):
		for i in range(len(players_pos)):
			if(players_pos[i].text != ""):
				center = [players_pos[i].contours[0]+players_pos[i].contours[2]//2,players_pos[i].contours[1]+players_pos[i].contours[2]//2]
				playerpos.append([center, players_pos[i].text])

	top_img = topview.create_topview(hg_matrix, playerpos, playerKalman[selectedPlayer].oocIndex)
	#获取当前时间
	now_seconds=int(Count /originalFps%60)
	now_minutes=int(Count/originalFps/60)
	total_second=int(frame_count /originalFps%60)
	total_minutes=int(frame_count/originalFps/60)
	#   { <参数序号> : <填充> <对齐）> <宽度> <,> <.精度> <类型>}.
	Time_now_vs_total="Time:{:>3}:{:>02}|{:>3}:{:0>2}".format(now_minutes,now_seconds,total_minutes,total_second)
	cv2.putText(top_img, Time_now_vs_total, (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示时间
	return playerpos, top_img, now_minutes, now_seconds