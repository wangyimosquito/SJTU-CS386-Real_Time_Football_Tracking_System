import cv2

range_width = 20
range_width_half = range_width // 2 

'''
提取目标轮轮廓或球员框质心附近的色调均值
'''

def is_team1_player(average_hue,hueColor):
    if average_hue in range(int(hueColor[0]) - range_width_half, int(hueColor[0]) + range_width_half):
        return True
    return False

def is_team2_player(average_hue,hueColor):
    if average_hue in range(int(hueColor[1]) - range_width_half, int(hueColor[1]) + range_width_half):
        return True
    return False

def is_team1_keeper(average_hue,hueColor):
    if average_hue in range(int(hueColor[2]) - range_width_half, int(hueColor[2]) + range_width_half):
        return True
    return False

def is_team2_keeper(average_hue,hueColor):
    if average_hue in range(int(hueColor[3]) - range_width_half, int(hueColor[3]) + range_width_half):
        return True
    return False

def is_ball(average_hue,hueColor):
    if average_hue in range(int(hueColor[4]) - range_width_half, int(hueColor[4]) + range_width_half):
        return True
    return False

def is_linesman(average_hue,hueColor):
    if average_hue in range(int(hueColor[5]) - range_width_half, int(hueColor[5]) + range_width_half):
        return True
    return False

def is_judge(average_hue,hueColor):
    if average_hue in range(int(hueColor[6]) - range_width_half, int(hueColor[6]) + range_width_half):
        return True
    return False

#物体色块中的平均色调
#通过将图像转换成HSV模式读取[色相，饱和度，明度]
# def average_hue(x, y, width, height, frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#hsv[[frame_height, frame_weight, [色相,色调,饱和度]]]
#     sum = 0
#     n_points = 0
#     for i in range(x, x + width + 1):
#         for j in range(y, y + height + 1):
#             if(j< 332 and i <1920):#存在越界问题，待解决
#                 sum += hsv[j, i, 0]#只提取色相即可
#             n_points += 1

#     return sum // n_points

def average_hue(x, y, width, height, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#hsv[[frame_height, frame_weight, [色相,色调,饱和度]]]
    sum = 0
    n_points = 0
    for i in range(3):
        for j in range(3):
            if(j+y-1< 332 and i+x-1 <1920):#存在越界问题，待解决
                sum += hsv[j+y-1, i+x-1, 0]#只提取色相即可
                n_points += 1

    return sum // n_points
