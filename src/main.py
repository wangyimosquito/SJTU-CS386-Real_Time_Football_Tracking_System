import numpy as np
import numpy.linalg as la
import os
import os.path
import cv2, sys
import matplotlib.pyplot as plt
import bgextraction
from firstframe import firstframe
from hueselector import HueSelector
import topview
from PyQt5 import QtWidgets

import playertrack
from pyqtMultithread2 import mainwin

bg_filpath = '../img/side-view.jpg'
vid_filepath = '../vid/panorama.avi'
hgmatrix_filepath = '../txt/hgmatrix.txt'
huedetect = '../txt/hue.txt'
playerrec = '..//txt//playerrec.txt'

def main():
	sys.setrecursionlimit(3000)
	if(not os.path.isfile(bg_filpath)):
		if(not os.path.exists('../img')):
			os.mkdir('../img')
		print ("Background has not been extracted, will extract.")
		bgextraction.extract_background(vid_filepath)
	print ("Background image found")

	if(not os.path.isfile(hgmatrix_filepath)):
		if(not os.path.exists('../txt')):
			os.mkdir('../txt')
		print ("Homography matrix has not been created.")
		topview.create_homography()
	hg_matrix = np.loadtxt(hgmatrix_filepath)

	if(not os.path.isfile(huedetect)):
		if(not os.path.exists('../txt')):
			os.mkdir('../txt')
		print("Hue of objects has not been selected.")
		HueSelector()
	print ("Hue Color Found.")

	if(not os.path.isfile(playerrec)):
		if(not os.path.exists('../txt')):
			os.mkdir('../txt')
		print("Playerrec of objects has not been selected.")
		firstframe()
	print ("Player Rec Found.")

	app = QtWidgets.QApplication(sys.argv)
	w = mainwin()
	w.show()
	sys.exit(app.exec_())

main()
