import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import time

start_time = time.time()
field_corners = np.empty([4,2], dtype = "float32")
field_counter = 0 


def field_click(event, x, y, flags, param):
	global field_counter
	if (event == cv2.EVENT_LBUTTONUP):
		if (field_counter >=4):
			print ("Press any key to continue")
		else:
			field_corners[field_counter, :] = [x,y]
			print (x,y)
			field_counter +=1

'''
手动标定球场四个角落以用于生成视角转换矩阵，以及生成球场俯视图
'''

def create_homography():
	global field_counter
	filename_topview = '..//img//top-view.jpg'
	filename_sideview = '..//img//side-view.jpg'
	hgcoord_filepath = '..//txt//hgmatrix.txt'
	
	top_image = cv2.imread(filename_topview)
	side_image = cv2.imread(filename_sideview)
	
	print ("Select the four corners from the Background")
	print ("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
	cv2.namedWindow('Side-View',cv2.WINDOW_FREERATIO)
	cv2.setMouseCallback('Side-View', field_click, None)
	cv2.imshow('Side-View', side_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	side_view_corners = np.copy(field_corners)
	np.savetxt('..//txt//sideviewcorners.txt',side_view_corners )
	top_view_corners = np.array([[44, 393], [44, 30], [598,30], [598, 393]], dtype  = "float32")

	H = cv2.findHomography(side_view_corners, top_view_corners)[0]
	np.savetxt(hgcoord_filepath, H)
	return H



def create_topview(hg_matrix, input_pts, selectedID):
	filename_topview = '../img/top-view.jpg'
	top_image = cv2.imread(filename_topview)
	pts = np.matrix(np.zeros(shape=(len(input_pts),3)))#[[x,y], type_of_obj]
	c = 0
	for i in input_pts:
		x,y = i[0][0], i[0][1]
		pts[c,:] = np.array([x,y,1], dtype = "float32")
		c+=1
	newPoints = np.empty([len(input_pts),3], dtype = "float32")
	c = 0
	for i in pts:
		newPoints = hg_matrix*(i.T)
		x = int(newPoints[0]/float(newPoints[2]))
		y = int(newPoints[1]/float(newPoints[2]))
		if(c == selectedID):
			cv2.circle(top_image,(x,y),6,(0,0,255),-1)#显示红色球员
			cv2.putText(top_image, input_pts[c][1], (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
		else:
			if(input_pts[c][1] != ""):
				cv2.circle(top_image,(x,y),6,(255,255,255),-1)#显示白色球员
				cv2.putText(top_image, input_pts[c][1], (x,y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255))#显示球员字号
		c +=1
	return top_image

#create_homography()
