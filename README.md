# SJTU-CS386-football-tracking
SJTU - CS386 Course Project, Football Player Tracking System

# Readme
### GUI example
![avatar](/img/GUI.png)

### Install

项目所需库

+ numpy
+ opencv3
+ PyQt5
+ matplotlib



### Usage

##### Run the testcase

The testcase in this repository is `vid/panorama.avi`。To run the tracking system, click `exe/main.exe`，then click `Play` button for real-time tracking and analysis.

This system has no pause function for it is designed to be a real-time tracking system. The backend won't store previous analysis results。But in the `txt/posistion data` preserves the 20 players' position records in the testcase video.

##### Quit the System

Click the `Quit` button for quiting. Close the window directly will cause the next running of this system to preserve the last running speed, running distance and running map records. But if you meant to preserve these, click the close button at up right.

##### Export the Data

对于手动选取的球员技术指标记录，在点击`Export Data`按钮后会自动生成`txt/playerAction.txt`数据记录。球员跑动路程图像，球员跑动平均速度图像存储于`项目工程文件/src/Speed.png`和`src/Distance.png`中，两队球员的跑动路径图像存储于`img/HotMapTeam1.png`以及`img/HotMapTeam2.png`中。

##### 更换视频源

本项目的比赛视频源可以进行更换。更换步骤如下所示：

1. 再次运行旧比赛视频，在中部数据控制面板选择`Clear Above Data`按钮即可清除旧视频相关的数据记录，点击`Quit`按钮进行退出。

2. 检查`txt`文件下记录是否清除完毕，若没有清除，可选择手动删除。

3. 检查`img`文件下是否清除side-view图片，若没有清除，可选择手动删除。

4. 在`src/main.py`中注释掉以下交互界面运行代码。

   ```python
   import playertrack
   from pyqtMultithread2 import mainwin
   ...
   app = QtWidgets.QApplication(sys.argv)
   w = mainwin()
   w.show()
   sys.exit(app.exec_())
   ```

5. 在`vid`中放入希望运行的源文件，命名为`panorama.avi`。

6. 在终端中运行main.py文件，此时脚本会依次执行背景提取，赛场四角手动标定，球员颜色手动标定，以及20名球员位置的手动框选。

   + 背景提取：脚本自动提取视频前500帧平均背景。
   + 赛场四角手动标定：按照左下，左上，右上，右下顺序依次点击赛场四角，选取完毕后按下任意键退出。
   + 球员颜色标定：选取两队球员和守门员躯干部份色调最明显部分，点击即可，选取完毕后按下任意键退出。
   + 20名球员位置的手动框选：手动框选20名球员的初始位置，前10名球员为一队，后10名球员为一队。手动框选后按下任意键，屏幕上即显示选框。20名球员框选完毕后按下任意键退出。

7. 以上准备工作完成后即可恢复注释的代码，再次运行交互界面。

