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

##### 运行报告测试用例视频追踪

项目报告中测试所用足球比赛原视频为`vid/panorama.avi`。若想运行追踪系统，则可直接点击`exe/main.exe`，再点击`Play`按钮即可开始实时分析和追踪。

本系统是实时分析系统，因此并不设置暂停功能。同时后台并不存储前次分析追踪结果。但在`txt/posistion data`文件中存储有测试用例20名球员的完整追踪坐标结果供参考分析。

##### 系统退出

退出实时分析系统时需要点击`Quit`按钮进行正确退出，若直接点击右上角关闭窗口，前次运行的速度，跑动路程，跑动路径等图像不会清除更新，但若想保留以上文件，可以点击右上角关闭窗口。

##### 导出数据

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

