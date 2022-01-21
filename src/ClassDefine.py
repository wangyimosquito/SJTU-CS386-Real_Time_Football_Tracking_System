import numpy as np
import cv2

class PlayerAction():
	def __init__(self,action, min, sec, ID):
		self.type = action
		self.sec = sec
		self.min =min
		self.playerID = ID

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
        self.oocIndex = -1
        self.w = 0
        self.h = 0
        self.speed = [0,0]
        self.totalDistant = 0
        self.displaySpeed = 0
        self.playerAction = list()
        self.lastUnMove = 0
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